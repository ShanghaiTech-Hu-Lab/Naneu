import torch
from typing import Sequence, Union
from torch import SymInt
from torch.fft import ifftshift, fftshift

from torchkbnufft import KbNufft, KbNufftAdjoint, calc_tensor_spmatrix, calc_density_compensation_function, KbInterp, KbInterpAdjoint
from einops import rearrange

from .. import NufftInterface
from ...fft.functional import itok

class TORCHKBNUFFTSampling2d(NufftInterface):
    scale_nufft: torch.Tensor
    scale_nuifft: torch.Tensor
    
    def __init__(self, image_size: Sequence[Union[int, SymInt]], traj: torch.Tensor, oversampling: float = 2, dtype = torch.float32,):
        """
        traj: (-0.5, 0.5)
        """
        super().__init__(image_size, traj)
        self.dtype = dtype
        self.oversampling = oversampling

        # Modules
        self._nufft_obj = KbNufft(im_size=image_size[-2:])
        self._inufft_obj = KbNufftAdjoint(im_size=image_size[-2:])
        self._grid_obj = KbInterpAdjoint(im_size=image_size[-2:], grid_size=image_size[-2:])
        self._degrid_obj = KbInterp(im_size=image_size[-2:], grid_size=image_size[-2:])

        self.register_buffer("scale_nufft" , torch.tensor(1.0))
        self.register_buffer("scale_nuifft", torch.tensor(1.0))

    def traj_normalize(self, traj: torch.Tensor) -> torch.Tensor:
        """
        normalize traj to (-pi, pi)
        """
        return traj / self.kmax * torch.pi

    def pre_calculate(self):
        dummy_image = torch.ones(*self.image_size, dtype=self.get_complex_type(self.dtype),device = self.traj.device)
        dummy_image = dummy_image.unsqueeze(0).unsqueeze(0)
        traj = self.traj_normalize(self.traj) # normalize traj to (-pi, pi)
        traj = rearrange(traj, 'phase readout pos -> pos (phase readout)')
        traj = torch.cat([torch.zeros_like(traj[:, 0:1]), traj], dim=1) # add center point of kspace to traj sequence

        # kbnufft routine
        kdata = self._nufft_obj(dummy_image, traj)
        kspace_center = kdata[...,0]
        self.scale_nuifft = dummy_image.sum() / kspace_center # update buffer
        kdata = kdata * self.scale_nufft
        kspace_center = kdata[...,0]
        interp_mats = calc_tensor_spmatrix(traj,im_size=self.image_size, table_oversamp=2)
        dcomp = calc_density_compensation_function(ktraj=traj, im_size=self.image_size)
        kdata = kdata * dcomp
        image = self._inufft_obj(kdata, traj, interp_mats)
        self.scale_nuifft = kspace_center / image.sum() # update buffer

    def grid(self, input: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        """
        input: [batch, channel, phase, readout]
        traj: [phase, readout, pos]

        Returns:
        kdata: [batch, channel, hight, wigth]
        """
        input_shape = input.shape
        p, r, _ = traj.shape
        traj = self.traj_normalize(traj)

        input = input.view(-1, *input_shape[-3:])
        input = rearrange(input, '... channel phase readout ->... channel (phase readout)')
        traj = rearrange(traj, 'phase readout pos -> pos (phase readout)')

        interp_mats = calc_tensor_spmatrix(traj,im_size=self.image_size, grid_size=self.image_size, table_oversamp=2)
        input = self._grid_obj(input, traj, interp_mats)

        input = input.view(*input_shape[:-3], *input.shape[-3:])
        input = fftshift(input, dim=(-2, -1))
        return input

    def dcomp(self, traj: torch.Tensor) -> torch.Tensor:
        p, r, _ = traj.shape
        traj = self.traj_normalize(traj)
        traj = rearrange(traj, 'phase readout pos -> pos (phase readout)')
        dcomp = calc_density_compensation_function(ktraj=traj, im_size=self.image_size)
        dcomp = rearrange(dcomp, '... (phase readout) ->... phase readout', phase=p, readout=r)
        dcomp = dcomp * self.scale_nuifft
        return dcomp

    def nufft(self, image: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        """
        image: [..., batch, channel, h, w]
        traj: [phase, readout, pos]
        """
        image_shape = image.shape
        p, r, _ = traj.shape
        traj = self.traj_normalize(traj)
        
        image = image.view(-1, *image_shape[-3:])
        traj = rearrange(traj, 'phase readout pos -> pos (phase readout)')
        kdata = self._nufft_obj(image, traj)
        kdata = kdata * self.scale_nufft
        kdata = rearrange(kdata, '... channel (phase readout) -> ... channel phase readout', phase=p, readout=r)
        kdata = kdata.view(*image_shape[:-3], *kdata.shape[-3:])
        return kdata
    
    def nuifft(self, kdata: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        """
        kdata: [..., batch, channel, phase, readout]
        traj: [phase, readout, pos]
        """
        kdata_shape = kdata.shape
        p, r, _ = traj.shape
        traj = self.traj_normalize(traj)

        kdata = kdata.view(-1, *kdata_shape[-3:])
        traj = rearrange(traj, 'phase readout pos -> pos (phase readout)')
        kdata = rearrange(kdata, '... channel phase readout ->... channel (phase readout)')
        interp_mats = calc_tensor_spmatrix(traj,im_size=self.image_size, table_oversamp=2)
        dcomp = calc_density_compensation_function(ktraj=traj, im_size=self.image_size)
        kdata = kdata * dcomp
        image = self._inufft_obj(kdata, traj, interp_mats)
        image = image * self.scale_nuifft
        image = image.view(*kdata_shape[:-3], *image.shape[-3:])
        return image
    
    def A(self, image: torch.Tensor) -> torch.Tensor:
        # image = image * self.csm
        kdata = self.nufft(image, self.traj)
        return kdata
    
    def At(self, kdata: torch.Tensor) -> torch.Tensor:
        image = self.nuifft(kdata, self.traj)
        # image = self.mulchan2single(image, self.csm)
        return image
    
    def get_phase(self, traj: torch.Tensor) -> torch.Tensor:
        # traj: [phase, readout, axis]
        phase = traj[:,-1,:] - traj[:,0,:]
        phase = torch.atan2(phase[:,1], phase[:,0])
        # phase: [phase]
        return phase

    def traj2mask(self, traj: torch.Tensor = None) -> torch.Tensor:
        if traj is None:
            traj = self.traj
        mask = torch.ones((1, 1, *traj.shape[:-1])).to(traj)
        mask = self.nuifft(mask, traj)
        mask = mask.view(*mask.shape[-2:])
        mask = itok(mask)
        mask = mask.abs()
        return mask

    def forward(self):
        pass