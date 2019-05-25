There are 2 ways to contribute to this repository:
1. Propose a new Feature Request and most probably it will be approved and you start working on it.
2. Hunt down bugs from existing issues

## Below is the how scratchai is structured

P.S. You can find other directories than the ones mentioned below.
Those are places of active development and completely unstable and 
its not concrete how it will interact with the entire library.

scratchai
  - nets
    - clf - contains implementations of classifiers
    - seg - contains implementations of segmentation model
  - DataLoaders
    - DataLoader.py - the Base Class for all 
    - ImgLoader.py - the file which is the base class for Image Files
    - SegLoader.py - the file which implements the Loader for Segmentation datasets
  - Learners
    - learner.py - the file which helps to train
  - imgutils.py - file to contain all image utility functions
  - version.py - stores version related information
  - attacks
    - attacks - Stores implementation of all available attacks
  - pretrained
    - urls.py - stores all the urls to the pretrained models in scratchai
    - README.md - stores details on how the pretrained models were obtained.
  
tests
   - test_nets.py     - ensures all modules are behaving in an expected fashion
   - test_learners.py - ensures learners are behaving in an expected fashion
   - test_attacks.py  - ensures all attacks are working okay.
   - test_utils.py    - ensures all utils functions are working as expected
   
   
### Add new style for real time style transfer

1. Add the style name say `xy` here in `avbl_style` list
https://github.com/iArunava/scratchai/blob/8fa93416e0c66e916d7df85cb1eba2c19dca6c1d/scratchai/one_call.py#L78

2. Add url to download the file from with varialble name `xy_url` in `pretrained/urls.py`
