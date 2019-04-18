"""
The metrics used to measure the performance of models.
"""

def miou(inp, lab):
  """
  Mean Intersection over Union (mIOU).

  Arguments
  --------
  inp : torch.tensor, [N x 3 x H x W]
        The original input images to the model.
  lab : torch.tensor, [N x H x W]
        The corresponding labels of the images.

  Returns
  -------
  miou : float
         The mean intersection over union value.
  
  Raises
  ------
  ValueError : If there is a shape mismatch.

  Notes
  -----
  Do note: if batches of data are passed it is necessary that
  inp - [N x 3 x H x W]
  lab - [N x H x W]

  where each matrix in lab is have each pixel value in the range of [0, C)
  where C is the number of classes.
  """
  
  dims = len(list(inp.shape))
  assert dims in [3, 4]
  # Assert batch_size, height and width matches
  if dim == 4: assert inp.shape[0] == lab.shape[0]
  assert inp.shape[-1] == lab.shape[-1]
  assert inp.shape[-2] == lab.shape[-2]
  
  # TODO Complete the function

  return 0
