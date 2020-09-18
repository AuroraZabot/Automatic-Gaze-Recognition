#pragma once
#include "EyeDetection.h"

//------------------------------------------------------------------------------------
// IMAGE LOADING
//------------------------------------------------------------------------------------
void EyeDetection::load(String& imageFolder, vector<Mat>& loadedImages)
{
	vector<String> filenames;
	glob(imageFolder, filenames);

	for (int i = 0; i < filenames.size(); ++i)
	{
		Mat img = imread(filenames[i], IMREAD_COLOR);
		loadedImages.push_back(img);
	}
}

//------------------------------------------------------------------------------------
// PREPROCESSING
//------------------------------------------------------------------------------------
void EyeDetection::intensityEqualization(Mat& input, Mat& output)
{
	cvtColor(input, output, COLOR_BGR2GRAY);

	vector<Mat> grayChannels(3);
	split(output, grayChannels);

	equalizeHist(grayChannels[0], grayChannels[0]);

	merge(grayChannels, output);
	cvtColor(output, output, COLOR_GRAY2BGR);
}

void EyeDetection::luminanceAdjustment(Mat& input, Mat& output)
{
	cvtColor(input, output, COLOR_BGR2YCrCb);

	vector<Mat> channelsYCbCr(3);
	split(output, channelsYCbCr);

	equalizeHist(channelsYCbCr[0], channelsYCbCr[0]);

	merge(channelsYCbCr, output);
	cvtColor(output, output, COLOR_YCrCb2BGR);
}

void EyeDetection::saturationEnhancement(Mat& input, Mat& output)
{
	cvtColor(input, output, COLOR_BGR2HSV);

	vector<Mat> hsvChannels(3);
	split(output, hsvChannels);

	hsvChannels[1] = hsvChannels[1] * 4;

	merge(hsvChannels, output);
	cvtColor(output, output, COLOR_HSV2BGR);
}

void EyeDetection::hsvEqualization(Mat& input, Mat& output)
{
	cvtColor(input, output, COLOR_BGR2HSV);

	vector<Mat> hsvChannels(3);
	split(output, hsvChannels);

	equalizeHist(hsvChannels[1], hsvChannels[1]);
	equalizeHist(hsvChannels[2], hsvChannels[2]);

	merge(hsvChannels, output);
	cvtColor(output, output, COLOR_HSV2BGR);
}

void EyeDetection::imageBlur(Mat& input, Mat& output, int kernelSize, double sigma, const char& kind, const char& selection)
{
	if (kind == 'g' && selection == 'b') //blurred images
		GaussianBlur(input, output, Size(kernelSize, kernelSize), sigma, sigma);

	else if (kind == 'g' && selection == 's') //sharpen images
	{
		GaussianBlur(input, output, Size(kernelSize, kernelSize), sigma, sigma);
		addWeighted(input, 1.5, output, -0.8, 0, output);
	}
	else if (kind == 'm' && selection == '-') //medial filter, maintain the edges
		medianBlur(input, output, kernelSize);
}

//------------------------------------------------------------------------------------
// EDGE & CIRCLE DETECTION
//------------------------------------------------------------------------------------
void EyeDetection::edgeDetection(Mat& input, Mat& output)
{
	//From the documentation the best results in Canny are achieved with a 2:1 or 3:1 regard to the threshold selection
	const double highThreshold = 200.0;
	const double lowThresold = highThreshold * 0.5;
	const int apertureSize = 3;

	Canny(input, output, lowThresold, highThreshold, apertureSize, false);
}

void EyeDetection::morphOperation(Mat& input, Mat& output)
{
	//The closure operation consists in first dilate then erode the objects, in order to "close" the gaps
	const int kernelSize = 3;
	Mat ellipse = getStructuringElement(MORPH_ELLIPSE, Size(kernelSize, kernelSize), Point(0, 0));

	morphologyEx(input, output, MORPH_CLOSE, ellipse, Point(1, 1), 1, BORDER_CONSTANT);
}

void EyeDetection::houghCircle(Mat& edgeImage, vector<Vec3f>& circles)
{
	//Since I need a B&W image, I make a check on the number of channels
	if (edgeImage.channels() == 3)
		cvtColor(edgeImage, edgeImage, COLOR_BGR2GRAY);

	//The parameters are choosen in the following way: the first parameter is related to the accuracy: the more high it is the more false circle we
	//detect; but if we take something too small, it would have problem in the detection
	double accumulator = 2.0;
	//Since we want just one detection on the images, we'll take the min distance pretty high
	double minDist = 50.0;
	//The high threshold from the documentation should have the same value of the canny one; if we take the smaller threshold too high, we would ù
	//not detect circles
	const double highThreshold = 200.0;
	const double lowThreshold = 10.0;
	//Considering the dimension of the pupil, I set a small radius in the detection, also to speed up the computation
	const int minRadius = 1;
	const int maxRadius = 9;

	HoughCircles(edgeImage, circles, HOUGH_GRADIENT, accumulator, minDist, highThreshold, lowThreshold, minRadius, maxRadius);
}

//------------------------------------------------------------------------------------
// FACE & EYES LOCALIZATION
//------------------------------------------------------------------------------------
void EyeDetection::eyeExtraction(Mat& input, vector<Mat>& finalEyes)
{
	//Equalization of the image
	Mat grayImage;
	intensityEqualization(input, grayImage);

	//In order to better recognize eyes in the face region, we would sharp a bit the image
	Mat sharpImage;
	const int kernelSize = 5;
	const double sigma = 150;
	const char kind = 'g';
	const char selection = 's';

	imageBlur(input, sharpImage, kernelSize, sigma, kind, selection);

	lbpCascade(input, grayImage, sharpImage, finalEyes);
}

void EyeDetection::lbpCascade(Mat& originalImage, Mat& grayImage, Mat& sharpImage, vector<Mat>& finalEyes)
{
	CascadeClassifier frontalface_cascade("lbpcascade_frontalface.xml");
	CascadeClassifier profileface_cascade("lbpcascade_profileface.xml");

	//Detect faces through LBP Cascade
	Mat dummyImage = originalImage.clone();
	vector<Rect> frontalFaces, profileFaces;

	//The parameters are choosen such that the image is resized around 1.05 at each scale and how many neighbors the rectangle should have in
	frontalface_cascade.detectMultiScale(grayImage, frontalFaces, 1.05, 3);
	profileface_cascade.detectMultiScale(grayImage, profileFaces, 1.05, 3);

	//If there's no frontal faces, check if there's a profile one
	if (!frontalFaces.empty())
	{
		for (int i = 0; i < frontalFaces.size(); ++i)
		{
			Mat faceRect = sharpImage(frontalFaces[i]);

			haarCascade(originalImage, dummyImage, frontalFaces[i], faceRect, finalEyes);
		}
	}
	else
	{
		for (int i = 0; i < profileFaces.size(); ++i)
		{
			Mat faceRect = sharpImage(profileFaces[i]);

			haarCascade(originalImage, dummyImage, profileFaces[i], faceRect, finalEyes);
		}
	}

	if (frontalFaces.empty() && profileFaces.empty())
	{
		cout << "Face not correctcly detected" << endl;
		return;
	}

	imshow("Eye Detection", dummyImage);
}

void EyeDetection::haarCascade(Mat& image, Mat& dummyImage, Rect& facesRect, Mat& faceImage, vector<Mat>& eyesRect)
{
	CascadeClassifier eyes_cascade("haarcascade_eye.xml");

	vector<Rect> eyes, refinedEyes;
	eyes_cascade.detectMultiScale(faceImage, eyes, 1.05, 2);

	//If there are too many rectangles, refine the computation
	if (eyes.size() > 2)
		eyesRefinements(eyes, refinedEyes);
	else
		refinedEyes = eyes;

	//Store all the rectangles containing the eyes such that in the following I can choose the one with the highest area
	vector<Mat> eye(refinedEyes.size());
	for (int j = 0; j < refinedEyes.size(); ++j)
	{
		eye[j] = faceImage(refinedEyes[j]);

		Point center(facesRect.x + refinedEyes[j].x + refinedEyes[j].width / 2, facesRect.y + refinedEyes[j].y + refinedEyes[j].height / 2);
		int radius = cvRound((refinedEyes[j].width + refinedEyes[j].height) * 0.25);
		circle(dummyImage, center, radius, Scalar(20, 20, 196), 2);
	}

	//Select just one eye, in particular the one with the highest area; this is just a method to speed up a little bit the computation
	eyeSelection(eye, eyesRect);
}

void EyeDetection::eyesRefinements(vector<Rect>& inputEyes, vector<Rect>& refinementEyes)
{
	vector<int> inputEyesArea(inputEyes.size());
	const int epsilon = 200;
	const int theta = 4000;

	//Sort the array such that it is sorted by increasing area
	sort(inputEyes.begin(), inputEyes.end(), [](const Rect& a, const Rect& b) {return a.area() < b.area();});

	for (int t = 0; t < inputEyes.size(); ++t)
		inputEyesArea[t] = inputEyes[t].area();

	//If two rectangles differs too much, they're probably not eyes (they have almost the same size)
	for (int t = 1; t < inputEyes.size(); ++t)
	{
		if (inputEyesArea[t] - inputEyesArea[t - 1] < epsilon)
		{
			refinementEyes.push_back(inputEyes[t - 1]);
			refinementEyes.push_back(inputEyes[t]);
		}
	}

	//If all the rectangles are too much different, take the two with the highest area, since statistically the 
	//two other elements mismatched are nostrils or mouth contours; I've considered also the possibility there's a
	//nostril and an eye in the detection: in this case, take just the biggest area
	if (refinementEyes.empty())
	{
		int max = inputEyes.size() - 1;
		if (inputEyesArea[max] - inputEyesArea[max - 1] < theta)
		{ 
			refinementEyes.push_back(inputEyes[max - 1]);
			refinementEyes.push_back(inputEyes[max]);
		}
		else
			refinementEyes.push_back(inputEyes[max]);
	}
}

void EyeDetection::eyeSelection(vector<Mat>& eyeCouples, vector<Mat>& singleEye)
{
	//Take just the element with the larger area; from how we've obatined the eyes
	//surely the vector has a size <= 2.
	if (eyeCouples.size() == 0)
		return;
	else if (eyeCouples.size() == 1)
		singleEye.push_back(eyeCouples[0]);
	else
	{
		int area0 = eyeCouples[0].rows * eyeCouples[0].cols;
		int area1 = eyeCouples[1].rows * eyeCouples[1].cols;

		if (area0 > area1)
			singleEye.push_back(eyeCouples[0]);
		else
			singleEye.push_back(eyeCouples[1]);
	}
}

//------------------------------------------------------------------------------------
// COLOR ESTIMATION
//------------------------------------------------------------------------------------
void EyeDetection::colorDetection(vector<Mat>& eyeROI)
{
	//Initialization of the parameters for k-means and image enhancement
	int kernelSize = 3;
	double sigma = 200;
	const char kind = 'g';
	const char selection = 'b';

	int clusters = 5;
	int attempts = 5;
	int flag = KMEANS_PP_CENTERS;

	vector<Mat> dummyImage(eyeROI.size());

	for (int t = 0; t < eyeROI.size(); ++t)
	{
		imageBlur(eyeROI[t], dummyImage[t], kernelSize, sigma, kind, selection);
		enhanceEye(dummyImage[t], dummyImage[t]);
		kMeans(dummyImage[t], clusters, attempts, flag);
	}
}

void EyeDetection::enhanceEye(Mat& inputImage, Mat& enhancedImage)
{
	int height = inputImage.rows;
	int width = inputImage.cols;
	
	//If it's not too small, select a smaller portion of the eye
	Mat eyeSelection;
	if (height < 10 || width < 10)
		eyeSelection = inputImage;
	else
	{
		Rect ROI(width / 3, height / 3, width / 2.5, height / 3);
	    eyeSelection = inputImage(ROI);
	}

	//Adjust the luminance component
	luminanceAdjustment(eyeSelection, eyeSelection);

	//Enhance the saturation one
	saturationEnhancement(eyeSelection, enhancedImage);
}

void EyeDetection::kMeans(Mat& inputImage, int& clusters, int& attempts, int& flag)
{
	//Reshape the image into one single row input of data in order to initialize the method
	Mat data;
	inputImage.convertTo(data, CV_32F);
	data = data.reshape(1, data.total());

	Mat labels, centers;
	//Algorithm stops if we reach a certain accuracy/max number of iterations
	TermCriteria criteria { TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0 };
	kmeans(data, clusters, labels, criteria, attempts, flag, centers);

	//Round the center components
	centers.convertTo(centers, CV_8UC1);

	//I don't care to output the resulting image, I just need the centers BGR components
	colorGrasp(centers);
}

void EyeDetection::colorGrasp(Mat& centers)
{	
	int rows = centers.rows;
	int cols = centers.cols;

	//Split the centers into their BGR components
	Vec3b centersMatrix;
	vector<int> B, G, R;
	int blue, green, red;

	for (int i = 0; i < rows; ++i)
	{
		centersMatrix = centers.row(i);
		blue = centersMatrix[0];
		green = centersMatrix[1];
		red = centersMatrix[2];

		//Select components that stand out compared to the other ones
		if ((blue > 120 && green > 120 && red > 120))
		{
			B.push_back(centersMatrix[0]);
			G.push_back(centersMatrix[1]);
			R.push_back(centersMatrix[2]);
		}
	}

	colorPlot(B, G, R);
}

void EyeDetection::colorPlot(vector<int>& B, vector<int>& G, vector<int>& R)
{
	int blue, green, red;

	if (B.empty()) // && G.empty() && R.empty(), they've the same size of course
		cout << "Black/dark brown eyes" << endl;
	else
	{
		vector<int> sum(B.size());
		int t = 0;

		for (int k = 0; k < B.size(); ++k)
		{
			sum[k] = B[k] + G[k] + R[k];
		}

		if (sum.size() != 1)
		{
			for (int k = 0; k < sum.size(); ++k)
			{
				int max = 0;
				if (sum[k] > max)
				{
					max = sum[k];
					t = k;
				}
			}
		}

		blue = B[t];
		green = G[t];
		red = R[t];

		//Choose the color output proportionally to the highest peak
		if (blue > green)
		{
			if (blue > red)
				cout << "Blue eyes" << endl;
			else
				cout << "Brown eyes" << endl;
		}
		else
		{
			if (green > red)
				cout << "Green eyes" << endl;
			else
				cout << "Brown eyes" << endl;
		}
	}
}

//------------------------------------------------------------------------------------
// GAZE DETECTION
//------------------------------------------------------------------------------------
void EyeDetection::gazeEstimation(vector<Mat>& eyes)
{
	for (int i = 0; i < eyes.size(); ++i)
	{
		//We focus on a little portion of the eyes passed in input
		Mat eyeROI;
		int width = eyes[i].cols;
		int height = eyes[i].rows;

		Rect ROI(width / 3, height / 3, width / 2, height / 3);
		eyeROI = eyes[i](ROI);
		
		vector<Point> pupilLocation;

		pupilDetection(eyeROI, pupilLocation);
		pupilLocalization(eyeROI, pupilLocation);
	}
}

void EyeDetection::pupilDetection(Mat& eyeROI, vector<Point>& pupilLocation)
{
	Mat dummy = eyeROI.clone();

	//We equalize the saturation and the brightness components in the HSV space
	hsvEqualization(eyeROI, eyeROI);

	//Initialization of the parameters for the kernel (median blur)
	int kernelSize = 3;
	double sigma = 200;
	const char kind = 'g';
	const char selection = 's';

	imageBlur(eyeROI, eyeROI, kernelSize, sigma, kind, selection);

	//Parameter for the mask: we would maintain only dark components. I've checked with different initializations
	//and this one is the most suitable one
	Mat maskSelection;
	Scalar lowerBound = Scalar(0, 0, 0);
	Scalar upperBound = Scalar(95, 95, 95);

	//We apply the mask and threshold with Otsu algorithm in order to make the image binary
	inRange(eyeROI, lowerBound, upperBound, maskSelection);
	threshold(maskSelection, maskSelection, 245, 255, THRESH_OTSU);
	//morphOperation(maskSelection, maskSelection);

	//Here we would apply Canny Edge Detector to find edges; it makes the computation a little bit slower than other methods but
	//it is really accurate in the edge description, so I would use it anyway
	edgeDetection(maskSelection, maskSelection);
	
	//I call the method to obtain the contour image with ellipses
	Mat contourImage;
	contourPupil(maskSelection, contourImage);

	//We call Hough Transform to find circles into the image (the pupil) and add it to the output vector
	vector<Vec3f> pupil;
	houghCircle(contourImage, pupil);

	for (int t = 0; t < pupil.size(); ++t)
	{
		Point center(pupil[t][0], pupil[t][1]);
		pupilLocation.push_back(center);
	}
}

void EyeDetection::contourPupil(Mat& processedImage, Mat& contourImage)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(processedImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<RotatedRect> minRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());

	//I select just the components that look more similar to circles 
	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(Mat(contours[i]));
		if (contours[i].size() > 4)
		{
			RotatedRect temporaryEllipse = fitEllipse(Mat(contours[i]));
			if (abs(temporaryEllipse.size.height - temporaryEllipse.size.width) < 5)
				minEllipse[i] = temporaryEllipse;
		}
	}

	contourImage = Mat::zeros(processedImage.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		drawContours(contourImage, contours, i, Scalar(255, 255, 0), 1, 8, vector<Vec4i>(), 0, Point());
		ellipse(contourImage, minEllipse[i], Scalar(255, 255, 0), 1, 8);
	}
}

void EyeDetection::pupilLocalization(Mat& eyes, vector<Point>& pupils)
{
	//Division of the image into regions and corresponding gaze detected
	int width = eyes.cols;

	if (pupils.empty())
	{
		cout << "Pupil doesn't detect correctly" << endl;
		return;
	}

	if (pupils[0].x < (width / 4))
		cout << "Look left" << endl;
	else if (pupils[0].x > (width - width / 3))
		cout << "Look right" << endl;
	else
		cout << "Look straight" << endl;
}
