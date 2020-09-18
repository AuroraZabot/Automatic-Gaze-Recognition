#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;

class EyeDetection
{
public:
	//--------------------------------------------------------------------------------------------------------------------
	// IMAGE LOADING
	//--------------------------------------------------------------------------------------------------------------------
	//Load images from the folder defined in the constant string inside the method
	//--------------------------------------------------------------------------------------------------------------------

	//@input: folder inside the project
	//@output: loaded images
	void load(String& imageFolder, vector<Mat>& loadedImages);

	//--------------------------------------------------------------------------------------------------------------------
	// FACE & EYES LOCALIZATION
	//--------------------------------------------------------------------------------------------------------------------
	//Here I would use LBP and Haar Cascades in order to grasp at the first step faces (with LBP) then eyes (with Haar). 
	//Since there are a lot of mismatching due to the lack of accuracy of this kind of cascades, I need to do a bit of 
	//preprocessing on the images, using the methods I've implemented below, in particular the histogram equalization 
	//and the image sharpening. In the report I've put all the description of parameters I've choosen in order to obtain 
	//better results.
	//--------------------------------------------------------------------------------------------------------------------

	//This portion of the code is just a dummy method to recall all the others; it has the main aim to obtain all the 
	//kind of filtered images used in the total computation
	//@input: BGR image
	//@output: vector containing the portion of the face containing the eye (I've output just the one of the two with
	// the highest area, in order to speed up the computation)
	void eyeExtraction(Mat& input, vector<Mat>& finalEyes);

	//--------------------------------------------------------------------------------------------------------------------
	// COLOR DETECTION
	//--------------------------------------------------------------------------------------------------------------------
	//Here I try to grasp the color eye; I firstly work a bit on the image, blurring it and then enhancing luminance and
	//saturation components; then I cut a bit more the portion of the eyes, in order to crop out the eyebrows and the 
	//ruddiness. After this, I run k-means algorithm; I actually don't care about how the image is plotted, I just need
	//centers in order to grasp the color components of clusters (in this way I saved a bit of time). The following step
	//consider the fact that, canceled out all the darker components, with an high probability the only that survived are
	//the gray of the eyes, potentially the ruddiness and the iris component. The ruddiness has a kind of color that tend
	//to have a high red component but smaller ones in blue and green. So, we can crop out the component with this shape.
	//Finally, the last survived is the iris one. In this case, since we've enhance a bit the saturation of the image, 
	//probably the highest peak in the BGR component is the effective color of the eyes.
	//--------------------------------------------------------------------------------------------------------------------

	//As in the previous section, this is just a dummy method to recall the other ones; in this case it's used to 
	//preprocess the images before the real computation of the color, adjusting the luminance and saturating them.
	//@input: vector of eyes previously computed
	void colorDetection(vector<Mat>& eyeROI);

	//--------------------------------------------------------------------------------------------------------------------
	// GAZE DETECTION
	//--------------------------------------------------------------------------------------------------------------------
	//I just take the detected portion of the eye, estimate the location of the pupil and output its location in the ROI. 
	//The main idea is to find it considering if the pupil is close or not to the contour of the eye. In order to find 
	//the pupil I've thresolded the image with a black mask and then I compute edges; at the beginning I've also used
	//morphological operations of closing to improve the performances, but I've noticed that the output were worst in this
	//case. After this step, I searched contours in order to find patterns that look elliptic, in particular with the
	//lenghts almost similar (pupils are circular shaped). At the end, I use the Hough Circle Transform to detect pupils. 
	//--------------------------------------------------------------------------------------------------------------------

	//Another dummy method to call the various components of the final part of the project.
	//@input: vector of detected eyes
	void gazeEstimation(vector<Mat>& eyes);

private:
	//--------------------------------------------------------------------------------------------------------------------
	// PREPROCESSING
	//--------------------------------------------------------------------------------------------------------------------
	//Those methods are just useful tools to manage images, such histogram equalization, saturation enhance, luminance
	//adjustments and image filtering (gaussian blur/sharp and median blur); they have been used during the whole 
	//computation in various step, so I found useful to implement them separately for sake of clarity
	//--------------------------------------------------------------------------------------------------------------------

	//Histogram equalization of the gray image, I used it in particular in the face detection portion of the code;
	//@input: BGR image 
	//@output: equalizedImage in grayscale
	void intensityEqualization(Mat& input, Mat& output);

	//Luminance adjustment thorugh BGR to YCbCr conversion, with equalization of the first component, where the major 
	//portion of the information is located
	//@input: BGR image 
	//@output: equalizedImage in the luminance component
	void luminanceAdjustment(Mat& input, Mat& output);

	//Saturation enhancement, I've used it in particular in the eye color detection; I've just made a map from BGR to
	//HSV color space and enhance the second channel
	//@input: BGR image 
	//@output: saturated image
	void saturationEnhancement(Mat& input, Mat& output);

	//HSV color space equalization, in particular of the Brightness and Saturation components; I've used it in 
	//particular in the gaze detection part of the code
	//@input: BGR image
	//@output: equalized image in HSV
	void hsvEqualization(Mat& input, Mat& output);

	//In this method I just filter the image; I can choose to use a median blur or a gaussian through the "kind",
	//while, looking at the "selection", I can switch from a blurred effect (with a 
	//gaussian filter) or to a sharped one (using a weight function that makes the lowpass filter an highpass one).
	//This last selection is just referred to the gaussian filter
	//@input: BGR image
	//@input: kernel and sigma, main parameters to define the filter I'm applying
	//@input: kind, character that select gaussian or median filter
	//@input: selection, character that select if I need a blurred filter or a sharper one
	//@output: filtered image
	void imageBlur(Mat& input, Mat& output, int kernelSize, double sigma, const char& kind, const char& selection);

	//--------------------------------------------------------------------------------------------------------------------
	// EDGE & CIRCLE DETECTION
	//--------------------------------------------------------------------------------------------------------------------
	//This section contains useful tools I've used in particular in the pupil detection portion of the code; in particular
	//I've focussed on the Canny Edge detector since is very accurate and even though we add a bit of computational 
	//complexity is not so important for this kind of implementation; then I've implemented a morphological filter with
	//round kernel to develop the "closure" morphological operation and, finally, I computer Hough Circular Transform 
	//to detect the pupil.
	//--------------------------------------------------------------------------------------------------------------------

	//"Canny Edge Detector"; this choise was made since this is one of the most accurate methods to find edges; even
	//though we add a bit of computational complexity, for our purposes (not real time applications), it's not so
	//heavy
	//@input: image, normally smoothed with a gaussian or median filter
	//@output: edge image
	void edgeDetection(Mat& input, Mat& output);

	//Morphological operation that compute first the dilation, then the erosion, normally used to "close" open edges
	//@input: image, normally thresholded or edge
	//@output: rearranged image
	void morphOperation(Mat& input, Mat& output);

	//We would use Hough Circle since it is very effective in founding centers, in particular if the images are 
	//a bit manipulated. We can also set a lot of parameters in order to make the association more accurate
	//@input: image (must be converted in grayspace)
	//@output: vector of circles found
	void houghCircle(Mat& edgeImage, vector<Vec3f>& circles);

	//--------------------------------------------------------------------------------------------------------------------
	// FACE & EYES LOCALIZATION
	//--------------------------------------------------------------------------------------------------------------------

	//Here I've implemented LBP cascade to face detection; with respect to Haar cascade, the LBP solution normally has
	//more mismatching, but most of the time includes all the faces in the image; since we need to find eyes in 
	//particular and not faces, I've preferred this solution with refinements in a second moment to the eye detection
	//@input: BGR image, the equalized and the sharped one
	//@output: vector of final eyes passed by the Haar detection
	void lbpCascade(Mat& originalImage, Mat& grayImage, Mat& sharpImage, vector<Mat>& finalEyes);

	//At this point I preferred rely to Haar cascade on the portion of faces previously detected, instead of LBP, in 
	//order to obtain more accurate images; the detection worked quite well, but it presented some mismatching as well
	//@input: BGR image and the "dummy" one, used to draw circles on the detected eyes
	//@input: portion of the face in which we search the eyes (Rect and Mat, in this case it's used in particular to
	// drawing purposes
	//@output: vector of final eyes after refinements and selection
	void haarCascade(Mat& image, Mat& dummyImage, Rect& facesRect, Mat& faceImage, vector<Mat>& eyesRect);

	//Since after the first Haar cascade there would be some mismatching in the eyes detection (i.e. double detection 
	//on the same eye, nostrils or mouth), I've made some refinements considering areas, in order to detect properly 
	//eyes on each image. In particular, I've considered that eyes should have almost the same area, while the other 
	//detections in the image should have a smaller one, with high probability
	//@input: full vector containing eyes portions
	//@output: vector containing the survived eyes after the computation
	void eyesRefinements(vector<Rect>& inputEyes, vector<Rect>& refinementEyes);

	//This code is just to select one eye in case of double detection in order to speed up a bit the computation. 
	//The selection is made just considering the bigger area and pass it to the final vector
	//@input: vector of each couple of eyes
	//@output: selection of just one single eye
	void eyeSelection(vector<Mat>& eyeCouples, vector<Mat>& singleEye);

	//--------------------------------------------------------------------------------------------------------------------
	// COLOR ESTIMATION
	//--------------------------------------------------------------------------------------------------------------------

	//In this method I recall two of the preprocessing tools I've already speak about, in order to bright and saturate 
	//the image; if the eye dimension is too small, the ROI is untouched, otherwise we select a smaller portion of the 
	//eye in order to reach easily the iris
	//@input: portion of the eye
	//@output: preprocessed image
	void enhanceEye(Mat& inputImage, Mat& enhancedImage);

	//This method just compute K-Means. The image is reshaped into one single row input of data following the method 
	//description. The algorithm stops after a certain number of iterations or when a certain level of accuracy is reached. 
	//At the end I didn't output the final result but actually I just call the method to plot the final color of the iris
	//@input: the enhanced image
	//@input: all the parameters I need to initialize the algorithm, such number of cluster, number of attempts and how
	// the initial centers are found
	void kMeans(Mat& inputImage, int& clusters, int& attempts, int& flag);

	//Here I just captured centers and divided them into the respective BGR components
	//@input: centroids obtained from k-means
	void colorGrasp(Mat& centers);

	//With this final method I just output the color of eyes that I've grasped; I took the highest peak in the BGR 
	//components, since with the saturation we enhance those ones.
	//@input: vectors of the BGR components
	void colorPlot(vector<int>& B, vector<int>& G, vector<int>& R);

	//--------------------------------------------------------------------------------------------------------------------
	// GAZE DETECTION
	//--------------------------------------------------------------------------------------------------------------------

	//This method just detect the pupil using a mask that isolate darker components, a threshold to level everything at 
	//the same color with Otsu's method and edge detection though Canny. Then, at the end, call the Hough Circle Transform 
	//to detect the pupil
	//@input: area of the eye to detect
	//@output: center of the pupil located
	void pupilDetection(Mat& eyesROI, vector<Point>& pupilLocation);

	//Here I just track the contour of the image given the preprocessed input. At the output I would obtain an image in
	//which I select some elliptic shapes iff their lenghts are similar, in order to have major probabilities of selecting
	//the pupil (that has a circular shape)
	//@input: edge image
	//@output: contourned image
	void contourPupil(Mat& processedImage, Mat& contourImage);

	//Here we just output the gaze estimated looking at the pupil detection. We just find if the pupil is close to the
	//margin of the eye or not
	//@input: location of the eye
	//@input: pupils
	void pupilLocalization(Mat& eyes, vector<Point>& pupils);
};