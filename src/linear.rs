use sliced::{
    custos::{number::Float, Alloc, Device, IsShapeIndep, OnNewBuffer},
    Buffer, GemmMayGrad, Matrix, RandOp, RowOpMayGrad,
};

pub struct Linear<'a, T, D: Device, const I: usize, const O: usize> {
    weights: Matrix<'a, T, D>,
    bias: Matrix<'a, T, D>,
}

impl<'a, T: Float, D: Device + OnNewBuffer<T, D>, const I: usize, const O: usize>
    Linear<'a, T, D, I, O>
{
    pub fn new(device: &'a D) -> Self
    where
        D: RandOp<T> + Alloc<T>,
    {
        let mut weights = Matrix::new(device, I, O).require_grad();
        device.rand(&mut weights, T::from_f64(-0.1), T::from_f64(0.1));
        // let mut weights = Matrix::from((device, I, O, vec![T::from_f64(0.01); I * O])).require_grad();
        // device.rand(&mut weights, -T::one() / T::two(), T::one() / T::two());
        //let mut weights = Matrix::from((device, I, O, vec![T::one(); I*O]));

        Linear {
            weights,
            bias: Matrix::new(device, 1, O).require_grad(),
        }
    }

    #[inline]
    pub fn forward(&self, inputs: &Matrix<'a, T, D>) -> Matrix<'a, T, D>
    where
        D: GemmMayGrad<T> + RowOpMayGrad<T>,
    {
        //inputs.gemm(&self.weights).add_row(&self.bias)
        let mut out = inputs.gemm(&self.weights);
        out.add_row_mut(&self.bias);
        out
    }

    pub fn params<'b>(&'b mut self) -> Vec<Param<'b, 'a, T, D>>
    where
        D: IsShapeIndep,
    {
        vec![Param::new(&mut self.weights), Param::new(&mut self.bias)]
    }
}

pub struct Param<'a, 'b, T, D: Device> {
    pub param: &'a mut Buffer<'b, T, D>,
}

impl<'a, 'b, T, D: Device> Param<'a, 'b, T, D> {
    pub fn new(param: &'a mut Buffer<'b, T, D>) -> Self {
        Param { param }
    }
}
