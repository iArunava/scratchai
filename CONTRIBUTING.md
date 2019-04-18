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
  
tests
   - test_nets - ensures all modules are behaving in an expected fashion
   - test_learners - ensures learners are behaving in an expected fashion
   
   
