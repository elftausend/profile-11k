use std::vec;

use graplot::{Scatter, XEnd, BLUE, GREEN, RED, VIOLET};
use linear::Linear;
use loss::{cce, cce_grad};
use sgd::SGD;
use sliced::{
    custos::{Autograd, Base, Cached, Cursor, TapeActions},
    Matrix, Mean, Onehot, CPU,
};

mod linear;
mod loss;
mod sgd;

fn main() {
    let device = CPU::<Autograd<Cached<Base>>>::new();

    let first_one_x = -0.33;
    let mut data = vec![first_one_x; 30];
    data.append(
        vec![
            first_one_x - 0.08,
            first_one_x - 0.04,
            first_one_x + 0.04,
            first_one_x + 0.08,
        ]
        .as_mut(),
    );
    data.append(vec![first_one_x - 0.035, first_one_x - 0.07, first_one_x - 0.105].as_mut());

    let end_first_one = data.len();

    let second_one_x = -0.10;
    data.append(vec![second_one_x; 30].as_mut());
    data.append(
        vec![
            second_one_x - 0.08,
            second_one_x - 0.04,
            second_one_x + 0.04,
            second_one_x + 0.08,
        ]
        .as_mut(),
    );
    data.append(
        vec![
            second_one_x - 0.035,
            second_one_x - 0.07,
            second_one_x - 0.105,
        ]
        .as_mut(),
    );

    let end_second_one = data.len();

    let k = 0.06;
    data.append(vec![k; 30].as_mut());
    data.append(vec![k + 0.035, k + 0.07, k + 0.105, k + 0.14, k + 0.175].as_mut());
    data.append(
        vec![
            k + 0.035,
            k + 0.07,
            k + 0.105,
            k + 0.14,
            k + 0.175,
            k + 0.21,
            k + 0.245,
            k + 0.28,
        ]
        .as_mut(),
    );

    let end_k = data.len();

    data.extend_from_slice(&[0.3, 0.1, -0.3, -0.26, -0.04, 0.11, 0.15, 0.125, -0.22]);

    let xs = Matrix::from((&device, 1, data.len(), data));

    let mut ys = Matrix::from((&device, 1, xs.len(), vec![0.; xs.len()]));

    for (i, y) in ys[..30].iter_mut().enumerate() {
        *y = (i as f32 * 0.1) - 1.5;
    }

    for y in ys[30..34].iter_mut() {
        *y = -1.5;
    }

    for (i, y) in ys[34..end_first_one].iter_mut().enumerate() {
        *y = ((i + 2) as f32 * -0.1) + 1.5;
    }

    for (i, y) in ys[end_first_one..end_first_one + 30].iter_mut().enumerate() {
        *y = (i as f32 * 0.1) - 1.5;
    }

    for y in ys[end_first_one + 30..end_first_one + 34].iter_mut() {
        *y = -1.5;
    }

    for (i, y) in ys[end_first_one + 34..end_second_one]
        .iter_mut()
        .enumerate()
    {
        *y = ((i + 2) as f32 * -0.1) + 1.5;
    }

    // k

    for (i, y) in ys[end_second_one..end_second_one + 30]
        .iter_mut()
        .enumerate()
    {
        *y = (i as f32 * 0.1) - 1.5;
    }

    for (i, y) in ys[end_second_one + 30..end_second_one + 35]
        .iter_mut()
        .enumerate()
    {
        *y = (i as f32 * 0.1) - 0.5;
    }
    for (i, y) in ys[end_second_one + 35..end_k].iter_mut().enumerate() {
        *y = (i as f32 * -0.15) - 0.5;
    }

    ys[end_k..].copy_from_slice(&[0.3, 0.1, -0.3, -0.26, -0.04, -1., -1.06, -1.1, 0.89]);

    let mut lin1 = Linear::<f32, _, 2, 32>::new(&device);
    let mut lin2 = Linear::<f32, _, 32, 4>::new(&device);

    let mut inputs = Vec::with_capacity(xs.len() * 2);
    for (x, y) in xs.read().iter().copied().zip(ys.read().iter().copied()) {
        inputs.push(x);
        inputs.push(y);
    }

    let mut classes = vec![0f32; end_first_one];
    classes.append(&mut vec![1.; end_second_one - end_first_one]);
    classes.append(&mut vec![2.; end_k - end_second_one]);
    classes.append(&mut vec![3.; xs.len() - end_k]);

    let samples = xs.len();

    let inputs = Matrix::from((&device, samples, 2, inputs));
    let classes = Matrix::from((&device, samples, 1, classes));
    let classes = device.onehot(&classes);
    let classes = Matrix::from((classes, samples, 4)).no_grad();

    let sgd = SGD { lr: 0.1 };
    for epoch in device.range(0..38000) {
        unsafe {
            device.gradients_mut().unwrap().zero_grad();
        };

        let out = lin1.forward(&inputs).relu();
        let out = lin2.forward(&out).softmax();

        let loss = cce(&out, &classes, out.cols());
        let grad = cce_grad(&out, &classes, out.rows());

        let avg_loss = device.mean(&loss);
        println!("epoch: {epoch}, loss: {avg_loss}");
        out.backward_with(&grad);

        sgd.step(lin1.params());
        sgd.step(lin2.params());
    }

    let mut scatter = Scatter::new((&xs.read()[..end_first_one], &ys.read()[..end_first_one]));
    scatter.plot.line_desc[0].color = RED;

    let mut scatter_green = Scatter::new((
        &xs.read()[end_first_one..end_second_one],
        &ys.read()[end_first_one..end_second_one],
    ));
    scatter_green.plot.line_desc[0].color = GREEN;
    scatter.add(scatter_green.plot);

    let mut scatter_blue = Scatter::new((
        &xs.read()[end_second_one..end_k],
        &ys.read()[end_second_one..end_k],
    ));
    scatter_blue.plot.line_desc[0].color = BLUE;
    scatter.add(scatter_blue.plot);

    let mut frags = Scatter::new((&xs.read()[end_k..], &ys.read()[end_k..]));
    frags.plot.line_desc[0].color = VIOLET;
    scatter.add(frags.plot);

    let colors = [RED, GREEN, BLUE, VIOLET];

    scatter.plot.desc.end_x = Some(XEnd(0.44));
    // scatter.plot

    for x in -95..90 {
        // -45..40
        let x = x as f32 / 100.;
        for y in -270..270 {
            // -170..170
            let y = y as f32 / 100.;
            let input = Matrix::from((&device, 1, 2, vec![x, y]));
            let out = lin1.forward(&input).relu();
            let out = lin2.forward(&out).softmax();

            let mut max = out[0];
            let mut idx = 0;
            for i in 1..out.len() {
                if out[i] > max {
                    max = out[i];
                    idx = i;
                }
            }
            let mut color = colors[idx];
            color.a = 0.015;

            let mut scatter2 = Scatter::new((vec![x], vec![y]));
            scatter2.plot.line_desc[0].color = color;
            scatter.add(scatter2.plot);
        }
    }

    scatter.show();
}
