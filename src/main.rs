use std::vec;

use graplot::Scatter;
use sgd::SGD;
use sliced::{custos::{Base, Cursor, TapeActions}, Matrix, Onehot, CPU};

mod linear;
mod loss;
mod sgd;

fn main() {
    let device = CPU::<Base>::new();

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
    for (i, y) in ys[end_second_one + 35..].iter_mut().enumerate() {
        *y = (i as f32 * -0.15) - 0.5;
    }

    let mut inputs = Vec::with_capacity(xs.len() * 2);
    for (x, y) in xs.read().iter().copied().zip(ys.read().iter().copied()) {
        inputs.push(x);
        inputs.push(y);
    }

    let mut classes = vec![0f32; end_first_one];
    classes.append(&mut vec![1.; end_second_one - end_first_one]);
    classes.append(&mut vec![2.; xs.len() - end_second_one]);

    let samples = inputs.len();

    let inputs = Matrix::from((&device, samples, 2, inputs));
    let classes = Matrix::from((&device, samples, 1, classes));
    let classes = device.onehot(&classes);
    let classes = Matrix::from((classes, samples, 3)).no_grad();


    let sgd = SGD { lr: 0.1 };
    for epoch in device.range(0..50) {
        unsafe {
            device.gradients_mut().unwrap().zero_grad();
        };

    }
    

    let mut scatter = Scatter::new((xs.read(), ys.read()));
    scatter.show();
}
