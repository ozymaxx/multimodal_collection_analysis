# Analysis of data collected from multimodal data collection studies
This repository contains source codes for analyzing spoken language and sketched symbols collected in multimodal data collection studies.

## Directory structure
There are two main folders here:

* **sketched_symbols**: This folder includes a Python script for analyzing sketched symbols. Note that you need to install [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/), [IDM feature extractor](https://github.com/ozymaxx/sketchfe), [NumPy](http://www.numpy.org/), [SciPy](https://scipy.org/) and [Matplotlib](https://matplotlib.org/) to run the script in this directory. Before running the script, make sure you copied `python` folder inside the LibSVM distribution with the source code on the main directory to `sketched_symbols` folder. Once the script has finished working, you will see a window with a plot showing accuracies for each class. 
* **speech_nl**: There are two sub-directories in this directory:
  * **analyze**: There are 7 scripts, each running analysis of different algorithms. For `deep_cnn.py`, `deep_rnn.py` and `deep_charbased.py`, you need to install [PyTorch](pytorch.org). Also make sure that your GPU has CUDA parallelization support. To use `fasttext_bigram.py` and `fasttext_unigram.py`, you need to download and `make` [FastText](https://github.com/facebookresearch/fastText) here. To run `knn_bow.py`, please install [Sklearn](scikit-learn.org/).
  * **report**: Once you created the confusion matrices, move them here. `plot_results.py` will output 3 plots (precision, recall and f-rates for each class) using the confusion matrices.

**A small note**: You should modify the following lines in scripts inside `speech_nl` directory to update paths of the data collected.

```
TEST_PHRASE_FILE_NAME = "analysis_corpus/testphrases_our.txt"
TRAIN_PHRASE_FILE_NAME = "analysis_corpus/trainphrases_our.txt"
VAL_PHRASE_FILE_NAME = "analysis_corpus/valphrases_our.txt"
TRAIN_VAL_PHRASE_FILE_NAME = "analysis_corpus/trainvalphrases_our.txt"
```

One more line for `knn_bow.py`:
```
STEMS_FILE_NAME = "stems_rej.txt"
```

For `analyze.py` inside `sketched_symbols`:

```
generic_path = '../soccer_annotated_sketches/'

classes=['player_motion','ball_motion','player_position1','player_position2']
class_counts = {}
class_correct_counts = {}

for cl in classes:
	class_counts[cl] = 0
	class_correct_counts[cl] = 0

files_plmovement = [join(generic_path+'playermotion/sketch/',f) for f in listdir(generic_path+'playermotion/sketch/') if isfile(join(generic_path+'playermotion/sketch/',f))]
files_ballmovement = [join(generic_path+'ballmotion/sketch/',f) for f in listdir(generic_path+'ballmotion/sketch/') if isfile(join(generic_path+'ballmotion/sketch/',f))]
files_plposition = [join(generic_path+'player/sketch/',f) for f in listdir(generic_path+'player/sketch/') if isfile(join(generic_path+'player/sketch/',f))]
files_otherplposition = [join(generic_path+'player_opposite/sketch/',f) for f in listdir(generic_path+'player_opposite/sketch/') if isfile(join(generic_path+'player_opposite/sketch/',f))]
```

Please feel free to open issues if you have problems about running the scripts. I'll try to get back as quickly as possible.

## See Also
To understand the architecture of `deep_cnn`,`deep_rnn`, please visit: [https://arxiv.org/abs/1603.03827](https://arxiv.org/abs/1603.03827). I used the ones shown in *Figure 1* (for feed-forward layers I used the leftmost one in *Figure 2*).

To see the architecture of `deep_charbased`, please visit: [https://pdfs.semanticscholar.org/b0ac/a3e7877c3c20958b0fae5cbf2dd602104859.pdf](https://pdfs.semanticscholar.org/b0ac/a3e7877c3c20958b0fae5cbf2dd602104859.pdf).

## Credits
Ozan Can Altıok - [Koç University Intelligent User Interfaces Laboratory](http://iui.ku.edu.tr) - oaltiok15 at ku dot edu dot tr

## Citation
If you'd like to use this work in your research, please cite [this article](https://iui.ku.edu.tr/sezgin_publications/2017/SezginAltiok-IUI-2017.pdf). Here's the BibTeX code for citing in LaTeX templates:

```
@inproceedings{altiok2017characterizing, title={Characterizing user behavior for speech and sketch-based video retrieval interfaces}, author={Alt{\i}ok, Ozan Can and Sezgin, Tevfik Metin}, booktitle={Proceedings of the Symposium on Non-Photorealistic Animation and Rendering}, pages={10}, year={2017}, organization={ACM} }
```
