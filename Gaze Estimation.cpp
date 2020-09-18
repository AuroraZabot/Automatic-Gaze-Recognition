#include <iostream>
#include "EyeDetection.h"

int main()
{
	EyeDetection eyeDetection;
	vector<Mat> images;
	String folderName;

	//Folder name is requested
	cout << "Enter the name of the folder inside the Project one where the images are located: " << endl;
	getline(cin, folderName);

	//Load of the images from a folder inside the project one
	eyeDetection.load(folderName, images);

	//Check on the images in input
	if (images.empty())
	{
		cerr << "The folder you looking at is empty or doesn't exist" << endl;
		return -1;
	}

	//Running the eye detection, color and gaze estimation
	for (int i = 0; i < images.size(); ++i)
	{
	    vector<Mat> eyes;
		eyeDetection.eyeExtraction(images[i], eyes);

		cout << "-------------------" << endl;
		cout << "Image number " << i << endl;
		cout << "-------------------" << endl;

		cout << "Color:" << endl;
		eyeDetection.colorDetection(eyes);

		cout << "-------------------" << endl;

		cout << "Gaze estimated:" << endl;
		eyeDetection.gazeEstimation(eyes);

		cout << "-------------------" << endl;
		cout << " " << endl;

		waitKey(0);
		destroyAllWindows();
	}

	return 0;
}

