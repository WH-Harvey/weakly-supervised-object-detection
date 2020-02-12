1. Install all relevant packages
pip install -r requirements.txt

2. All codes run by jupyter notebook
./example_images contains some example images
./data contains training and test data(It is too big to write on DVD and upload). The data could be downoload by http://host.robots.ox.ac.uk/pascal/VOC/voc2007/.`
./thesis contains the latex and pdf thesis
./example_reults contains the results of some examples

3. For convience, I make each demo for each approach
demo_cross.ipynb detects object with cross filter
demo_sum.ipynb detects object with sum filter
demo_surf.ipynb detects object with surf

All demos could directly run and the output is generated in ./tmp

4. Model.py contains the function about the network

5. utils.py contains all function for object detection

6. feature map visualize.ipynb is used to visualize feature maps of each channel. The output is stored in ./feature/n (n is the current layer).

7. feature map visualize multichannels.ipynb is used to visualize feature maps of all channels. The output is stored in ./all.

8. detection.ipynb is used to generate the output of test data for evaluation. The output is stored in ./eval/2. ./eval contains an sample

9. eval.ipynb is used to calculate the mAP. The output is stored in ./output. ./output contain an sample.

10. transfer_learning.ipynb is used to retrain a model for my dataset. The weights is stored in ./weights.
