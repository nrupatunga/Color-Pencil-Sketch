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
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>

#pragma comment(lib, "opencv_core2413d.lib") // core functionalities
#pragma comment(lib, "opencv_highgui2413d.lib") //GUI
#pragma comment(lib, "opencv_imgproc2413d.lib") // Histograms, Edge detection

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  main
 *  Description:  
 * =====================================================================================
 */
int main ( int argc, char *argv[] )
{
	//uchar au8Data[9] = {255, 0, 149, 0, 255, 196, 220, 0, 244};
	String strImgPath = "6--32.jpg";
	Mat sInputImgBGR = imread(strImgPath, IMREAD_COLOR);
	Mat sOutputImgBGR; Mat sInputImgYUV; Mat sOutputImgYUV;

	cvtColor(sInputImgBGR, sInputImgYUV, COLOR_BGR2YUV); 
	vector<Mat> YUVIn(3);
	vector<Mat> YUVOut(3);
	split(sInputImgYUV, YUVIn);
	//Mat sInputImg = Mat(Size(3,3), CV_8UC1, au8Data, Mat::AUTO_STEP);
	Mat sOutputImgGray;

	ColorSketch sColorSketch;
	sColorSketch.Generate(YUVIn[0], sOutputImgGray);

	YUVOut[0] = sOutputImgGray;
	YUVOut[1] = YUVIn[1];
	YUVOut[2] = YUVIn[2];
	merge(YUVOut, sOutputImgYUV);
	cvtColor(sOutputImgYUV, sOutputImgBGR, COLOR_YUV2BGR);

	namedWindow("Input", WINDOW_NORMAL);
	namedWindow("Input Gray", WINDOW_NORMAL);
	namedWindow("Output Sketch", WINDOW_NORMAL);
	namedWindow("Output Color Sketch", WINDOW_NORMAL);
	imshow("Input", sInputImgBGR);
	imshow("Input Gray", YUVIn[0]);
	imshow("Output Sketch", sOutputImgGray);
	imshow("Output Color Sketch", sOutputImgBGR);

	imwrite("outputgraysketch.png", sOutputImgGray);
	imwrite("outputcolorsketch.png", sOutputImgBGR);

	waitKey(0);
	return EXIT_SUCCESS;
} /* ----------  end of function main  ---------- */
