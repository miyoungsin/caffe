#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <stdio.h>
using namespace cv;
using namespace std;

int channel=3;
double result=0;

Mat convolution(int rowsize, int colsize, double ***a,int s,int f, int p, double **b, Mat re_image, Mat image)
{
    Mat re1_image(rowsize,colsize,image.type());
    Mat kernel = (Mat_<double>(3,3)<<-1,-1,-1,-1,8,-1,-1,-1,-1);

    filter2D(image,re1_image,image.depth(),kernel);

    //filter2D(image,re1_image,CV_16S,kernel);
    

    clock_t begin=clock();
    int x,y;
    double temp=0;

    for(int g=0; g<channel; g++)
    {
        x=0; y=0;
        for(int h=0; h<rowsize; h++)
        {
            for(int i=0; i<colsize; i++)
            {
                re_image.at<cv::Vec3b>(h,i)[g]=0;
                for(int j=0; j<f; j++)
                {
                    for(int k=0; k<f; k++)
                    {
                        temp+=b[j][k]*a[g][x+j][y+k];
                       
                    }         
                }
                 //printf("%f",temp);
                if(temp>=255) { re_image.at<cv::Vec3b>(h,i)[g]=255; }
                else if(temp<=0){ re_image.at<cv::Vec3b>(h,i)[g]=0; }           
                else re_image.at<cv::Vec3b>(h,i)[g]=(int)temp;
                //printf("%d %d ",re_image.at<cv::Vec3b>(h,i)[g], re1_image.at<cv::Vec3b>(h,i)[g]);
                //result += re1_image.at<cv::Vec3b>(h,i)[g] - re_image.at<cv::Vec3b>(h,i)[g];
                //result += result_image.at<cv::Vec3b>(h,i)[g];

                y+=s; temp=0;
                if(y+f-1>image.cols+(2*p)-1){ y=0; x+=s; }
            }
            //cout<<endl;
        }
    }//convolution
    cv::Mat result_image;
    //cv::subtract(re_image,re1_image,result_image);
    cv::subtract(re_image,re1_image,result_image,noArray(),-1);
    cv::namedWindow("result_image",CV_WINDOW_AUTOSIZE);
    cv::imshow("result_image",result_image);
    imwrite("con_ff.jpg",re1_image);

   

    for(int g=0; g<channel; g++)
    {
        x=0; y=0;
        for(int h=0; h<rowsize; h++)
        {
            for(int i=0; i<colsize; i++)
            {
                result += result_image.at<cv::Vec3b>(h,i)[g];
            }
        }
    }//convolution


    clock_t end=clock();
    double elapsed_secs=double(end-begin);
    cout<<"convolution time = "<<elapsed_secs<<endl; 

    imwrite("con.jpg",re_image);

    return re_image;
}
Mat MaxPooling(Mat p_image, Mat image, int p_rowsize, int p_colsize, int f, int s)
{
    clock_t begin=clock();
    int x,y;

    for(int g=0; g<channel; g++)
    {
        x=0; y=0;
        for(int h=0; h<p_rowsize; h++)
        {
            for(int i=0; i<p_colsize; i++)
            {
                p_image.at<cv::Vec3b>(h,i)[g]=0;
                for(int j=0; j<f; j++)
                {            
                    for(int k=0; k<f; k++)
                    {
                        if(image.at<cv::Vec3b>(x+j,y+k)[g] > p_image.at<cv::Vec3b>(h,i)[g]) { p_image.at<cv::Vec3b>(h,i)[g] = image.at<cv::Vec3b>(x+j,y+k)[g]; }
                    }             
                }
                if(p_image.at<cv::Vec3b>(h,i)[g]>255) { p_image.at<cv::Vec3b>(h,i)[g]=255; }
                else if(p_image.at<cv::Vec3b>(h,i)[g]<0){ p_image.at<cv::Vec3b>(h,i)[g]=0; }

                y+=s;
                if(y+f-1>image.cols-1){ y=0; x+=s; }

            }
        }
    }//max_pooling
    clock_t end=clock();
    double elapsed_secs=double(end-begin);
    cout<<"pooling time = "<<elapsed_secs<<endl; 

    imwrite("pool.jpg",p_image);

    return p_image;
}
void Activation(Mat img,int row, int col)
{
    for (int i=0; i<channel; i++)
    {
	    for (int j=0; j<row; j++)
        {
		    for (int k=0; k<col; k++)
            {
                img.at<cv::Vec3b>(j,k)[i] = 1/(1+exp(img.at<cv::Vec3b>(j,k)[i]));
            }
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
    Mat image;
    image = imread("tt.jpg",IMREAD_COLOR);
    if(image.empty())
    {
        cout<<"not image"<<endl;
        return -1;
    }
    namedWindow("original",WINDOW_AUTOSIZE);
    //imshow("original",image);
 
    int rowsize,colsize,p,s,w,f;
    int p_rowsize,p_colsize;

    cout<<"filter 행렬 값을 입력 F: ";
    cin>>f;

    double **b=new double *[f];
    for(int i=0; i<f; i++){b[i]=new double[f+1];}//filter 동적할당

    for(int i=0; i<f; i++)
    {
        for(int j=0; j<f; j++)
        {
             cin>>b[i][j];
        }
    }//filter

    cout<<"Padding 값과 stride값을 입력 P,S: ";
    cin>>p>>s;

    double ***a= new double **[channel];
    for(int i=0; i<channel; i++)
    {
        a[i]=new double *[image.rows+2*p];
        for(int j=0; j<image.rows+2*p; j++)
        {
            a[i][j]=new double [image.cols+2*p];
        }
    }

    for(int h=0; h<channel; h++)
    {
        for(int i=0; i<image.rows+2*p; i++)
        {
            for(int j=0; j<image.cols+2*p; j++)
            {
                if(i<p || j<p || i>image.rows+2*p-p-1 || j>image.cols+2*p-p-1) 
                { a[h][i][j]=255;}
                else a[h][i][j]=(double)image.at<cv::Vec3b>(i-p,j-p)[h];
            }
        }
    }//padding image

    rowsize=(image.rows-f+2*p)/s +1;
    colsize=(image.cols-f+2*p)/s +1;
    p_rowsize=(image.rows-f)/s +1;
    p_colsize=(image.cols-f)/s +1;

    Mat re_image(rowsize,colsize,image.type()); //convolution_result_image
    //Mat re1_image(rowsize,colsize,image.type()); //convolution_result_image
    Mat p_image(p_rowsize,p_colsize,image.type()); //pooling_result_image

    //convolution(rowsize,colsize,a,s,f,p,b,re_image,image);
    ////////////////////////////////////////////////////////////////////
    //Mat kernel = (Mat_<float>(3,3)<<0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625);

    //filter2D(image,re1_image,-1,kernel);
    //filter2D(image,re_image,image.depth(),kernel);
    convolution(rowsize,colsize,a,s,f,p,b,re_image,image);
    //imwrite("con_ff.jpg",re1_image);
    printf("%lf \n",result);

   
    ////////////////////////////////////////////////////////////////////
    //MaxPooling(p_image, image, p_rowsize, p_colsize, f, s);

    //Activation(convolution(rowsize,colsize,a,s,f,p,b,re_image,image),rowsize,colsize);
    //Activation(MaxPooling(p_image, image, p_rowsize, p_colsize, f, s),p_rowsize,p_colsize);
    //cv::Mat result_image;
    //cv::subtract(re_image,re1_image,result_image);
    //cv::subtract(re_image,re1_image,result_image,noArray(),-1);
    //cv::namedWindow("result_image",CV_WINDOW_AUTOSIZE);
    //cv::imshow("result_image",result_image);

    

   
    for(int i=0; i<channel; i++)
    {
        for(int j=0; j<image.rows+2*p; j++) { delete[]a[i][j]; }
        delete[]a[i];
    }
    delete[]a; 
    
   
    for(int i=0; i<f; i++){delete[]b[i];}
    delete[]b;

	waitKey(0);
    //return 0;
}
