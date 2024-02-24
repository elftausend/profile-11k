use sliced::{
    custos::{number::Float, AddGradFn, AddOperation, Combiner, Device},
    BinaryElementWise, Buffer, Clip, SumCols,
};

pub fn cce<'a, T, D>(
    preds: &Buffer<'a, T, D>,
    targets: &Buffer<T, D>,
    cols: usize,
) -> Buffer<'a, T, D>
where
    T: Float,
    D: Device + Clip<T, ()> + AddOperation + BinaryElementWise<T> + SumCols<T> + AddGradFn,
{
    let device = preds.device();
    // could use functions with grad fn -> use device.no_grad_ctx(|| { ... })
    // device.no_grad_ctx();
    let preds = device.clip(preds, T::as_generic(1E-7), T::as_generic(1. - 1E-7));
    let preds = device.mul(&preds, &targets);
    let preds = device.sum_cols(cols, &preds);
    device.apply_fn(&preds, |v| v.ln().neg())
}

pub fn cce_grad<'a, T: Float, D: Device + Clip<T, ()> + AddOperation + BinaryElementWise<T>>(
    preds: &Buffer<'a, T, D>,
    targets: &Buffer<T, D>,
    rows: usize,
) -> Buffer<'a, T, D> {
    let device = preds.device();
    let grad = device.div(targets, preds);
    device.apply_fn(&grad, move |v| v.neg().div(T::from_usize(rows)))
}
