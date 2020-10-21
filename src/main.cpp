#include <iostream>
#include <opencv2/opencv.hpp>

#include <random>
#include <functional>
#include "colors.h"

const cv::Size imgSize = {1000U, 1000U};
const uint16_t gridStep = 30U;
const cv::Size gridSize = {(imgSize.width / gridStep) + 1, (imgSize.height / gridStep) + 1};
const float scale = 0.01;
const uint16_t randRange = 10U;
const uint16_t rsComputingIterations = 3U;

std::random_device device;
typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<double> Distribution;
auto uniform_generator = std::bind(Distribution(0.0f, 1.0f), Engine(device()));

double random(const double range_min, const double range_max) {
  double ksi;
  #pragma omp critical ( random )
  {
    #if __cplusplus < 201103L
    ksi = (double)rand() / (double)RAND_MAX;
    #else
    ksi = static_cast<double>( uniform_generator());
    #endif
  }
  return ksi * (range_max - range_min) + range_min;
}

inline float base1(float r, float s) { return (1.f - r) * (1.f - s); }

inline float base2(float r, float s) { return (r) * (1.f - s); }

inline float base3(float r, float s) { return (r) * (s); }

inline float base4(float r, float s) { return (1.f - r) * (s); }

template<typename T, int cn>
cv::Vec<T, cn> getTquad(float r, float s, const cv::Vec<T, cn> &p1, const cv::Vec<T, cn> &p2, const cv::Vec<T, cn> &p3, const cv::Vec<T, cn> &p4) {
  return (p1 * base1(r, s)) +
         (p2 * base2(r, s)) +
         (p3 * base3(r, s)) +
         (p4 * base4(r, s));
}

cv::Vec2f computeRS(const cv::Point2f &p, const cv::Vec2f &p1, const cv::Vec2f &p2, const cv::Vec2f &p3, const cv::Vec2f &p4) {
  // [0]=r, [1]=s
  cv::Vec2f rsVector(0.5, 0.5);
  
  cv::Mat1f J(2, 2);
  cv::Mat1f J_inv(2, 2);
  cv::Vec2f T;
  
  for (int i = 0; i < 3; i++) {
    J.at<float>(0, 0) = (rsVector[1] - 1) * (p1[0] - p2[0]) + rsVector[1] * (p3[0] - p4[0]);
    J.at<float>(1, 0) = (rsVector[1] - 1) * (p1[1] - p2[1]) + rsVector[1] * (p3[1] - p4[1]);
    J.at<float>(0, 1) = (rsVector[0] - 1) * (p1[0] - p4[0]) + rsVector[0] * (p3[0] - p2[0]);
    J.at<float>(1, 1) = (rsVector[0] - 1) * (p1[1] - p4[1]) + rsVector[0] * (p3[1] - p2[1]);
    
    T = getTquad(rsVector[0], rsVector[1], p1, p2, p3, p4);
    
    cv::invert(J, J_inv, cv::DECOMP_SVD);
    
    cv::Vec2f TminusP = T - cv::Vec2f(p.x, p.y);
    cv::Mat1f temp = (J_inv * TminusP);
    
    rsVector = (rsVector - cv::Vec2f(temp(0), temp(1)));
  }
  return rsVector;
}

struct Quad {
  Quad(const cv::Point &p1, const cv::Point &p2, const cv::Point &p3, const cv::Point &p4) : p1(p1), p2(p2), p3(p3), p4(p4) {}
  
  const cv::Point p1;
  const cv::Point p2;
  const cv::Point p3;
  const cv::Point p4;
  
  std::vector<cv::Point> points;
  
  void debugDrawEdges(cv::Mat& mat){
    cv::line(mat, p1, p2, 0xff);
    cv::line(mat, p2, p3, 0xff);
    cv::line(mat, p3, p4, 0xff);
    cv::line(mat, p4, p1, 0xff);
  }
  
  void debugDraw(cv::Mat& mat){
    static int colorDebugCounter = 0;
    for(const auto& point : points){
      mat.at<cv::Vec3b>(point) = colors[colorDebugCounter];
    }
    colorDebugCounter++;
    colorDebugCounter %= colorsSize;
    
    cv::line(mat, p1, p2, 0xff);
    cv::line(mat, p2, p3, 0xff);
    cv::line(mat, p3, p4, 0xff);
    cv::line(mat, p4, p1, 0xff);
    //cv::imshow("Debug draw", mat);
    //cv::waitKey();
  }
  
  bool insideQuad(const cv::Point &p) {
    // TY HOVADO!
    return insideTriangle(p, p1, p2, p3) || insideTriangle(p, p3, p4, p1) || insideTriangle(p, p1, p4, p2)  || insideTriangle(p, p2, p4, p3);
  }
  
  static bool insideTriangle(const cv::Point &p, const cv::Point &p1, const cv::Point &p2, const cv::Point &p3) {
    int as_x = p.x - p1.x;
    int as_y = p.y - p1.y;
    
    bool s_ab = (p2.x - p1.x) * as_y - (p2.y - p1.y) * as_x > 0;
    
    if ((p3.x - p1.x) * as_y - (p3.y - p1.y) * as_x > 0 == s_ab) return false;
    
    if ((p3.x - p2.x) * (p.y - p2.y) - (p3.y - p2.y) * (p.x - p2.x) > 0 != s_ab) return false;
    
    return true;
  }
};

int main() {
  {
    const cv::Vec2f p1(1, 0);
    const cv::Vec2f p2(3, 0.25);
    const cv::Vec2f p3(4, 3);
    const cv::Vec2f p4(0, 3.5);
    const cv::Vec2f p(1.8, 2.7);
    
    auto rsVect = computeRS(p, p1, p2, p3, p4);
    
    std::cout << rsVect[0] << "  " << rsVect[1] << std::endl;
  }
  
  
  cv::Mat1b originalImage(imgSize, CV_8UC1);
  cv::Mat3b originalColorImage(imgSize, CV_8UC3);
  cv::Mat1b reconstructedImage(imgSize, CV_8UC1);
  cv::Mat3b reconstructedColorImage(imgSize, CV_8UC3);
  
  std::vector<Quad> quads;
  std::vector<cv::Point2i> gridPoints;
  
  
  gridPoints.reserve(gridSize.area());
  gridPoints.resize(gridSize.area());
  quads.reserve(gridSize.area());
  
  for (int row = 0; row < originalImage.rows; row++) {
    for (int col = 0; col < originalImage.cols; col++) {
      float val = cos(col * scale) * sin(row * scale);
      
      val += 1.0;
      val /= 2.0;
      val *= 255;
      
      originalImage.at<uint8_t>(row, col) = val;
      
      if (((row % gridStep) == 0) && ((col % gridStep) == 0)) {
        gridPoints.at((col / gridStep) * gridSize.width + (row / gridStep)) = cv::Point2i(col, row);
      }
    }
  }
  
  for (auto &point : gridPoints) {
    point.x += random(-randRange, randRange);
    point.y += random(-randRange, randRange);
    
    point.x = std::clamp(point.x, 0, imgSize.width - 1);
    point.y = std::clamp(point.y, 0, imgSize.height - 1);
  }
  
  for (int y = 0; y < gridSize.width; y++) {
    for (int x = 0; x < gridSize.height; x++) {
      const cv::Point &point = gridPoints.at(x * gridSize.width + y);
//      cv::circle(originalImage, point, 2, 0xff);
      
      cv::Point eastPoint;
      cv::Point southPoint;
      cv::Point southEastPoint;
      
      if ((x + 1) < gridSize.height) {
        eastPoint = gridPoints.at((x + 1) * gridSize.width + y);
      } else {
        eastPoint = point;
        eastPoint.x += gridStep;
      }
      
      if ((y + 1) < gridSize.width) {
        southPoint = gridPoints.at(x * gridSize.width + (y + 1));
      } else {
        southPoint = point;
        southPoint.y += gridStep;
      }
  
      if (((x + 1) < gridSize.height)&&((y + 1) < gridSize.width )){
        southEastPoint = gridPoints.at((x+1) * gridSize.width + (y + 1));
      }
      else{
        southEastPoint.y = southPoint.y;
        southEastPoint.x = eastPoint.x;
      }
      
      quads.emplace_back(Quad(point, southPoint, southEastPoint, eastPoint));
    }
  }
  
  for (int row = 0; row < originalImage.rows; row++) {
    for (int col = 0; col < originalImage.cols; col++) {
      for(Quad& quad : quads){
        cv::Point pt = cv::Point(col, row);
        if(quad.insideQuad(pt)){
          quad.points.emplace_back(pt);
  
          cv::Point2f ptf =static_cast<cv::Point2f>(pt);
          cv::Point2f p1f =static_cast<cv::Point2f>(quad.p1);
          cv::Point2f p2f =static_cast<cv::Point2f>(quad.p2);
          cv::Point2f p3f =static_cast<cv::Point2f>(quad.p3);
          cv::Point2f p4f =static_cast<cv::Point2f>(quad.p4);
          
          cv::Vec<uint8_t,1> val1(originalImage.at<uint8_t>(p1f));
          cv::Vec<uint8_t,1> val2(originalImage.at<uint8_t>(p2f));
          cv::Vec<uint8_t,1> val3(originalImage.at<uint8_t>(p3f));
          cv::Vec<uint8_t,1> val4(originalImage.at<uint8_t>(p4f));
          
          auto rsVect = computeRS(ptf, p1f, p2f, p3f, p4f);
          cv::Vec<uint8_t,1> val = getTquad(rsVect[0], rsVect[1], val1, val2, val3, val4);
  
          reconstructedImage.at<uint8_t>(pt) = val[0];
          
          break;
        }
      }
    }
  }
  
  
  for(Quad& quad : quads) {
    quad.debugDrawEdges(originalImage);
  }
  
  
  cv::imshow("originalImage", originalImage);
  cv::applyColorMap(originalImage, originalColorImage, cv::COLORMAP_JET);
  cv::imshow("originalColorImage", originalColorImage);
  
  
  cv::imshow("reconstructedImage", reconstructedImage);
  cv::applyColorMap(reconstructedImage, reconstructedColorImage, cv::COLORMAP_JET);
  cv::imshow("reconstructedColorImage", reconstructedColorImage);
  
  cv::waitKey();
  return 0;
  
}