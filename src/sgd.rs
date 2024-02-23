pub struct SGD<T> {
    pub lr: T,
}

use sliced::custos::{
    prelude::{ClearBuf, One, WriteBuf},
    Alloc, MayTapeActions, ZeroGrad,
};

use core::ops::{Mul, SubAssign};
use std::ops::{Deref, DerefMut};

use crate::linear::Param;

impl<T: Copy + One + Mul<Output = T> + SubAssign + 'static> SGD<T> {
    pub fn zero_grad<D>(&self, params: Vec<Param<T, D>>)
    where
        D: MayTapeActions + ZeroGrad<T> + WriteBuf<T> + Alloc<T> + ClearBuf<T> + 'static,
    {
        for param in params {
            param.param.grad_mut().clear();
        }
    }

    pub fn step<D>(&self, params: Vec<Param<T, D>>)
    where
        D: WriteBuf<T> + ZeroGrad<T> + Alloc<T> + MayTapeActions + 'static,
        D::Base<T, ()>: Deref<Target = [T]> + DerefMut,
    {
        for param in params {
            let grad = param.param.grad();
            for (value, grad) in param.param.iter_mut().zip(grad.iter()) {
                *value -= *grad * self.lr
            }
        }
    }
}
