use std::vec;

use graplot::Scatter;
use sliced::{custos::Base, Matrix, CPU};

mod linear;
mod loss;
mod sgd;

fn main() {
    let device = CPU::<Base>::new();

    let first_one_x = -0.33;
    let mut data = vec![first_one_x; 30]; 
    data.append(vec![first_one_x - 0.08, first_one_x - 0.04, first_one_x +0.04, first_one_x +0.08].as_mut());
    data.append(vec![first_one_x - 0.035, first_one_x - 0.07, first_one_x - 0.105].as_mut());


    let end_first_one = data.len();

    let second_one_x = -0.10;
    data.append(vec![second_one_x; 30].as_mut());
    data.append(vec![second_one_x - 0.08, second_one_x - 0.04, second_one_x +0.04, second_one_x +0.08].as_mut());
    data.append(vec![second_one_x - 0.035, second_one_x - 0.07, second_one_x - 0.105].as_mut());

    let end_second_one = data.len();
    
    let k = 0.06;
    data.append(vec![k; 30].as_mut());
    data.append(vec![k + 0.035, k + 0.07, k + 0.105, k + 0.14, k + 0.175].as_mut());
    data.append(vec![k + 0.035, k + 0.07, k + 0.105, k + 0.14, k + 0.175, k + 0.21, k + 0.245, k+0.28].as_mut());
    let mut xs = Matrix::from((
        &device, 1, data.len(),
        data
    ));

    // for x in xs.iter_mut() {
    //     *x /= 13.;
    // }

    let mut ys = Matrix::from((
        &device, 1, xs.len(),
        vec![
            0.; xs.len()
        ],
    ));


    for (i, y) in ys[..30].iter_mut().enumerate() {
        *y = (i as f32 * 0.1) - 1.5;
    }

    for y in ys[30..34].iter_mut() {
        *y = -1.5;
    }
    
    for (i, y) in ys[34..end_first_one].iter_mut().enumerate() {
        *y = ((i+2) as f32 * -0.1) + 1.5;
    }


    
    for (i, y) in ys[end_first_one..end_first_one+30].iter_mut().enumerate() {
        *y = (i as f32 * 0.1) - 1.5;
    }
    
    for y in ys[end_first_one+30..end_first_one+34].iter_mut() {
        *y = -1.5;
    }
    
    for (i, y) in ys[end_first_one+34..end_second_one].iter_mut().enumerate() {
        *y = ((i+2) as f32 * -0.1) + 1.5;
    }

    // k
    
    for (i, y) in ys[end_second_one..end_second_one+30].iter_mut().enumerate() {
        *y = (i as f32 * 0.1) - 1.5;
    }
    
    for (i, y) in ys[end_second_one+30..end_second_one+35].iter_mut().enumerate() {
        *y = (i as f32 * 0.1) - 0.5;
    }
    for (i, y) in ys[end_second_one+35..].iter_mut().enumerate() {
        *y = (i as f32 * -0.15) - 0.5;
    }


    let mut scatter = Scatter::new((xs.read(), ys.read()));
    scatter.show();
}
