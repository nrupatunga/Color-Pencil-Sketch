/*
* Copyright (c) 2003, 2016 Nrupatunga
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*
*/

#include "ColorSketch.hpp"
#include <numeric>
#include <functional>
#include <cmath>

#define DEBUG
#ifdef DEBUG
#pragma comment(lib, "opencv_core2413d.lib") // core functionalities
#pragma comment(lib, "opencv_highgui2413d.lib") //GUI
#pragma comment(lib, "opencv_imgproc2413d.lib") // Histograms, Edge detection
#else
#pragma comment(lib, "opencv_core2413.lib") // core functionalities
#pragma comment(lib, "opencv_highgui2413.lib") //GUI
#pragma comment(lib, "opencv_imgproc2413.lib") // Histograms, Edge detection
#endif

const float fKernel[8][5][5] = {
	//Kernel - 1 (0)
	{ { 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0 },
	{ 1, 1, 1, 1, 1 },
	{ 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0 } },
	//Kernel-2 (22.5)
	{ { 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 },
	{ 0.00000, 0.00000, 0.07612, 0.45880, 0.64760 },
	{ 0.23463, 0.61732, 1.00000, 0.61732, 0.23463 },
	{ 0.64760, 0.45880, 0.07612, 0.00000, 0.00000 },
	{ 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 } },
	//Kernel-3 (45)
	{ { 0.00000, 0.00000, 0.00000, 0.2574, 0.1716 },
	{ 0.00000, 0.00000, 0.29289, 1.00000, 0.2574 },
	{ 0.00000, 0.29289, 1.00000, 0.29289, 0.00000 },
	{ 0.2574, 1.00000, 0.29289, 0.00000, 0.00000 },
	{ 0.2574, 0.1716, 0.00000, 0.00000, 0.00000 } },
	//Kernel-4 (67.5)
	{{0.00000, 0.00000, 0.23463, 0.6476, 0.00000},
	{ 0.00000, 0.00000, 0.61732, 0.45880, 0.00000 },
	{ 0.00000, 0.07612, 1.00000, 0.07612, 0.00000 },
	{ 0.00000, 0.45880, 0.61732, 0.00000, 0.00000 },
	{ 0.00000, 0.64760, 0.23463, 0.00000, 0.00000 }},
	//Kernel-5 (90)
	{ { 0, 0, 1, 0, 0 },
	{ 0, 0, 1, 0, 0 },
	{ 0, 0, 1, 0, 0 },
	{ 0, 0, 1, 0, 0 },
	{ 0, 0, 1, 0, 0 } },
	//Kernel-6 (112.5)
	{ { 0.00000, 0.6476, 0.23463, 0.00000, 0.00000 },
	{ 0.00000, 0.45880, 0.61732, 0.00000, 0.00000 },
	{ 0.00000, 0.07612, 1.00000, 0.07612, 0.00000 },
	{ 0.00000, 0.00000, 0.61732, 0.45880, 0.00000 },
	{ 0.00000, 0.00000, 0.23463, 0.6476, 0.00000 } },
	//Kernel-7 (135)
	{ { 0.1716, 0.2574, 0.00000, 0.00000, 0.00000 },
	{ 0.2574, 1.00000, 0.29289, 0.00000, 0.00000 },
	{ 0.00000, 0.29289, 1.00000, 0.29289, 0.00000 },
	{ 0.00000, 0.00000, 0.29289, 1.00000, 0.2574 },
	{ 0.00000, 0.00000, 0.00000, 0.2574, 0.1716 } },
	//Kernel-8 (157.5)
	{{ 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 },
	{0.6476, 0.45880, 0.07612, 0.00000, 0.00000},
	{0.23463, 0.61732, 1.00000, 0.61732, 0.23463},
	{0.00000, 0.00000, 0.07612, 0.45880, 0.6476},
	{0.00000, 0.00000, 0.00000, 0.00000, 0.00000}}
};

//Initialization of Constant variables inside the class
ColorSketch::Params::Params():s32KernelSize(5), s32NumDirs(8){}

ColorSketch::ColorSketch()
{
	init(2);
}


ColorSketch::ColorSketch(int s32AmtDark)
{
	init(s32AmtDark);
}

void ColorSketch::init(int s32AmtDark)
{
	sSketchParams.s32AmtDark    = s32AmtDark;
}

void ColorSketch::Generate(Mat &sMatInput, Mat &sMatOutput)
{
	int s32NumChannels = sMatInput.channels();
	CV_Assert((s32NumChannels == 3) || (s32NumChannels == 1));

	if ( 3 == s32NumChannels ) {
		cvtColor(sMatInput, sMatInputGray, COLOR_BGR2GRAY);
	}else{
		sMatInputGray = sMatInput;
	}

	Mat sMatGradX, sMatGradXAbs, sMatGradY, sMatGradYAbs, sMatOutputImg;
	//X-gradient
	cv::Sobel(sMatInputGray, sMatGradX, CV_32FC1, 1, 0);
	//Y-gradient
	cv::Sobel(sMatInputGray, sMatGradY, CV_32FC1, 0, 1);
#if 1
	cv::Mat sMatOutputImgTemp = sMatGradX.mul(sMatGradX) + sMatGradY.mul(sMatGradY);
	cv::sqrt(sMatOutputImgTemp, sMatOutputImg);
	normalize(sMatOutputImg, sMatOutputImg, 0, 255, NORM_MINMAX);
	sMatOutputImg.convertTo(sMatOutputImg, CV_8UC1);
#else
	cv::convertScaleAbs(sMatGradX, sMatGradXAbs);
	cv::convertScaleAbs(sMatGradY, sMatGradYAbs);
	cv::addWeighted(sMatGradXAbs, 0.5, sMatGradYAbs, 0.5, 0, sMatOutputImg);
#endif

	//Line Drawing
	sMatLineSketch.create(Size(sMatInput.cols, sMatInput.rows), CV_8UC1);
	//Tone Map
	sMatToneMap.create(Size(sMatInput.cols, sMatInput.rows), CV_8UC1);
	//Final Sketc
	sMatFinalSketch.create(Size(sMatInput.cols, sMatInput.rows), CV_8UC1);

	GenerateLineDrawing(sMatInputGray, sMatLineSketch);
	GenerateToneMap(sMatInputGray, sMatToneMap);
	GenerateFinalSketch(sMatLineSketch, sMatToneMap, sMatFinalSketch);
	sMatOutput = sMatFinalSketch;
}

void ColorSketch::GenerateToneMap(const Mat &psMatInputGray, Mat &psMatToneMap)
{
	vector<double> vecHistOne(256);
	vector<double> vecCdfOne(256);
	vector<double> vecCdfTwo(256);

	// Initialize parameters for histogram calculation
	Mat          sMatHistTwo;
	int          s32HistSize = 256;        // bin size
	float        fRange[]    = { 0, 255 };
	const float *fRanges     = { fRange };

	GenerateSketchCurve(vecHistOne);
	partial_sum(vecHistOne.begin(), vecHistOne.end(), vecCdfOne.begin());

	//Histogram calculation of input
	calcHist(&psMatInputGray, 1, 0, Mat(), sMatHistTwo, 1, &s32HistSize, &fRanges);
	double dScale = cv::sum(sMatHistTwo)[0];
	sMatHistTwo   = sMatHistTwo/dScale;
	//cdf
	vecCdfTwo.at(0) = sMatHistTwo.at<float>(0, 0);
	for(int i = 1; i < sMatHistTwo.rows; i++){
		vecCdfTwo.at(i) = vecCdfTwo.at(i-1) + sMatHistTwo.at<float>(i, 0);
	}
	//ShowHistogram(sMatHistTwo);

	//Histogram matching
	vector<double> vecSub(256);
	vector<int> vecIndex(256);
	for ( int i = 0; i < vecCdfOne.size(); i++ ) {
		std::transform(vecCdfOne.begin(), vecCdfOne.end(), vecSub.begin(), std::bind2nd(std::minus<double>(), vecCdfTwo.at(i)));
		for ( int j = 0; j < vecSub.size(); j++ ) {
			vecSub[j] = std::abs(vecSub[j]);
		}
		std::vector<double>::iterator  itr = std::min_element(vecSub.begin(), vecSub.end());
		vecIndex[i] = itr - vecSub.begin();
	}

	for ( int i = 0; i < psMatInputGray.rows; i++ ) {
		const uchar *pu8InData  = psMatInputGray.ptr<uchar>(i);
		uchar *pu8OutData = psMatToneMap.ptr<uchar>(i);
		for ( int j = 0; j < psMatInputGray.cols; j++ ) {
			int s32Index;
			s32Index      = pu8InData[j];
			pu8OutData[j] = vecIndex.at(s32Index);
		}
	}
}

void ColorSketch::GenerateLineDrawing(const Mat &psMatInputGray, Mat &psMatLineSketch)
{
	int          s32Width      = psMatInputGray.cols;
	int          s32Height     = psMatInputGray.rows;
	const uchar *pu8RowDataOne = nullptr;
	const uchar *pu8RowDataTwo = nullptr;
	uchar       *pu8GradX      = nullptr;
	uchar       *pu8GradY      = nullptr;
	float       *pfGrad        = nullptr;

	Mat sMatLineSketch;
	//Gradient in X and Y direction
	Mat sMatGradX = Mat(Size(s32Width, s32Height), CV_8UC1, Scalar::all(0));
	Mat sMatGradY = Mat(Size(s32Width, s32Height), CV_8UC1, Scalar::all(0));
	Mat sMatGrad  = Mat(Size(s32Width, s32Height), CV_32FC1, Scalar::all(0));

	//
	Mat *psMatDirOut = new Mat[sSketchParams.s32NumDirs];
	Mat *psMatCmap   = new Mat[sSketchParams.s32NumDirs];
	Mat *psMatSketch = new Mat[sSketchParams.s32NumDirs];
	for ( int i = 0; i < sSketchParams.s32NumDirs; i++ ) {
		psMatCmap[i]   = Mat(Size(s32Width, s32Height), CV_32FC1, Scalar::all(0));
		psMatSketch[i] = Mat(Size(s32Width, s32Height), CV_32FC1, Scalar::all(0));
	}

	// Gradient in X and Y direction
	for (int i = 0; i < s32Height-1; i++) {
		pu8RowDataOne = psMatInputGray.ptr<uchar>(i);
		pu8RowDataTwo = psMatInputGray.ptr<uchar>(i+1);
		pu8GradX      = sMatGradX.ptr<uchar>(i);
		pu8GradY      = sMatGradY.ptr<uchar>(i);
		for (int j = 0; j < s32Width-1; j++) {
			pu8GradX[j] = std::abs(pu8RowDataOne[j + 1] - pu8RowDataOne[j]);
			pu8GradY[j] = std::abs(pu8RowDataTwo[j] - pu8RowDataOne[j]);
		}
	}

	// X gradient - Last row
	pu8RowDataOne = psMatInputGray.ptr<uchar>(s32Height-1);
	pu8GradX      = sMatGradX.ptr<uchar>(s32Height-1);
	for ( int j = 0; j < s32Width-1; j++ ) {
		pu8GradX[j] = std::abs(pu8RowDataOne[j + 1] - pu8RowDataOne[j]);
	}

	// Y Gradient - Last col
	for (int i = 0; i < s32Height-1; i++) {
		pu8RowDataOne = sMatInputGray.ptr<uchar>(i);
		pu8RowDataTwo = sMatInputGray.ptr<uchar>(i+1);
		pu8GradY      = sMatGradY.ptr<uchar>(i);

		pu8GradY[s32Width-1] = std::abs(pu8RowDataTwo[s32Width-1] - pu8RowDataOne[s32Width - 1]);
	}

	//Gradient = sqrt(gradientX*gradientX+gradientY*gradientY)
	for ( int i = 0; i<s32Height; i++ ) {
		pu8RowDataOne = sMatGradX.ptr<uchar>(i);
		pu8RowDataTwo = sMatGradY.ptr<uchar>(i);
		pfGrad        = sMatGrad.ptr<float>(i);
		for ( int j = 0; j<s32Width; j++ ) {
			pfGrad[j] = std::sqrt(pu8RowDataOne[j]*pu8RowDataOne[j] + pu8RowDataTwo[j]*pu8RowDataTwo[j]);
		}
	}

	// Release the memory not in use
	sMatGradX.release();
	sMatGradY.release();

	normalize(sMatGrad, sMatGrad, 0, 255, NORM_MINMAX);
	//sMatGrad.convertTo(sMatGrad, CV_8UC1);

	//Directional Kernel
	Mat sMatKernel = Mat(Size(sSketchParams.s32KernelSize, sSketchParams.s32KernelSize), CV_32FC1);
	Point sPtAnchor(sMatKernel.cols - sMatKernel.cols/2 - 1, sMatKernel.rows - sMatKernel.rows/2 - 1);

	for ( int i = 0; i < sSketchParams.s32NumDirs; i++ ) {
		for ( int j = 0; j < sSketchParams.s32KernelSize; j++ ) {
			for ( int k = 0; k < sSketchParams.s32KernelSize; k++ ) {
				sMatKernel.at<float>(j,k) = fKernel[i][j][k];
			}
		}
		filter2D(sMatGrad, psMatDirOut[i], -1, sMatKernel, Point(-1, -1), 0, BORDER_CONSTANT);
	}

	int s32W        = psMatDirOut[0].cols;
	int s32H        = psMatDirOut[0].rows;
	int s32MaxValue = -1;
	int s32MaxIndex = -1;

	for (int i = 0; i < s32H; i++) {
		for (int j = 0; j < s32W; j++) {
			for ( int k = 0; k < sSketchParams.s32NumDirs; k++ ) {
				if(psMatDirOut[k].at<float>(i,j) > s32MaxValue){
					s32MaxIndex = k;
					s32MaxValue = psMatDirOut[k].at<float>(i, j);
				}
			}
			psMatCmap[s32MaxIndex].at<float>(i, j) = sMatGrad.at<float>(i, j);
			s32MaxIndex = -1; s32MaxValue = -1;
		}
	}
	//Release the memory
	sMatGrad.release();

	for ( int i = 0; i < sSketchParams.s32NumDirs; i++ ) {
		for ( int j = 0; j < sSketchParams.s32KernelSize; j++ ) {
			for ( int k = 0; k < sSketchParams.s32KernelSize; k++ ) {
				sMatKernel.at<float>(j,k) = fKernel[i][j][k];
			}
		}
		filter2D(psMatCmap[i], psMatSketch[i], -1, sMatKernel, Point(-1, -1), 0, BORDER_CONSTANT);
	}
	add(psMatSketch[0], psMatSketch[1], psMatSketch[0]);
	add(psMatSketch[0], psMatSketch[2], psMatSketch[0]);
	add(psMatSketch[0], psMatSketch[3], psMatSketch[0]);
	add(psMatSketch[0], psMatSketch[4], psMatSketch[0]);
	add(psMatSketch[0], psMatSketch[5], psMatSketch[0]);
	add(psMatSketch[0], psMatSketch[6], psMatSketch[0]);
	add(psMatSketch[0], psMatSketch[7], psMatSketch[0]);

	sMatLineSketch = psMatSketch[0];
	normalize(sMatLineSketch, sMatLineSketch, 0, 255, NORM_MINMAX);
	sMatLineSketch = sMatLineSketch - 255;
	sMatLineSketch = (-1)*sMatLineSketch;
	normalize(sMatLineSketch, sMatLineSketch, 0, 255, NORM_MINMAX);
	sMatLineSketch.convertTo(sMatLineSketch, CV_8UC1);

	sMatLineSketch.copyTo(psMatLineSketch);
}

void ColorSketch::GenerateSketchCurve(vector<double> &vecHist)
{
	const float kPI = 3.1416;
	int s32Ua = 105, s32Ub = 225, s32Mud = 90, s32DeltaB = 9, s32DeltaD = 11, s32OmegaOne = 76, s32OmegaTwo = 18, s32OmegaThree = 6;
	double dDistOne = 0.0; double dDistTwo = 0.0; double dDistThree = 0.0; double dSumHist = 0.0;
	
	float fValue;
	for ( int i = 0; i < 256; i++ ) {
		if ( i < s32Ua || i > s32Ub ) {
			fValue = 0;
		}else{
			fValue = 1.0/(s32Ub - s32Ua);
		}
		dDistOne      = ( s32OmegaTwo * fValue );
		dDistTwo      = (( 1.0 / s32DeltaB ) * exp(-(255.0 - i) / s32DeltaB) ) * s32OmegaOne;
		dDistThree    = ( 1.0 /sqrt(2 * kPI * s32DeltaD) * exp(-(i - s32Mud) * (i - s32Mud) / (2.0 * s32DeltaD * s32DeltaD)) *  s32OmegaThree);
		vecHist.at(i) = dDistOne + dDistTwo + dDistThree;
		dSumHist      = dSumHist + vecHist.at(i);
	}

	for ( int i = 0; i < 256; i++ ) {
		vecHist.at(i) = vecHist.at(i)/dSumHist;
	}
}

#if 0
void ColorSketch::ShowHistogram(Mat &psMatHist)
{
	int nc = 1, bins = 256;
	const char* wname = { "gray"};
	Scalar colors[1] = {Scalar(255,0,0)};
	Mat canvas;
	Mat hist;

	normalize(psMatHist, hist, 0, 125, NORM_MINMAX);

	// Display each histogram in a canvas
	canvas = Mat::zeros(125, bins, CV_8UC1);

	for (int j = 0, rows = canvas.rows - 1; j < bins; j++)
	{
		Point pt1 = Point(j, rows);
		Point pt2 = Point(j, (rows - hist.at<float>(j)));
		line( canvas, pt1, pt2, Scalar(255), 1, 8, 0 );
	}

	imshow(wname, canvas);
	waitKey(0);
}
#endif

void ColorSketch::GenerateFinalSketch(const Mat &psMatLineSketch, const Mat &psMatToneMap, Mat &psMatFinalSketch)
{
	float fMaxLineSketch, fMaxToneMap;
	int s32Rows, s32Cols;

	fMaxLineSketch = *(std::max_element(psMatLineSketch.begin<uchar>(),psMatLineSketch.end<uchar>()));
	fMaxToneMap    = *(std::max_element(psMatToneMap.begin<uchar>(),psMatToneMap.end<uchar>()));

	s32Cols = psMatLineSketch.cols;
	s32Rows = psMatLineSketch.rows;

	for ( int i = 0; i < s32Rows; i++ ) {
		const uchar *pu8LineSketch  = psMatLineSketch.ptr<uchar>(i);
		const uchar *pu8ToneMap     = psMatToneMap.ptr<uchar>(i);
		uchar       *pu8FinalSketch = psMatFinalSketch.ptr<uchar>(i);
		for ( int j = 0; j < s32Cols; j++ ) {
			float fPixelValue = ( pu8LineSketch[j]*pu8ToneMap[j]*255.0)/(fMaxLineSketch*fMaxToneMap);
			pu8FinalSketch[j] = (uchar)fPixelValue;
		}
	}
}
