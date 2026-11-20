#include "fastsaminference.h"
#include <QDebug>

FastSamInference::FastSamInference(QObject *parent) : QObject(parent)
{
}

FastSamInference::~FastSamInference()
{
}

bool FastSamInference::loadModel(const QString &modelPath)
{
    try {
        _net = cv::dnn::readNetFromONNX(modelPath.toStdString());
        if (_net.empty()) {
            emit logMessage("❌ Failed to load FastSAM ONNX model.");
            return false;
        }
        
        // Use CPU for safety and compatibility
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        _isModelLoaded = true;
        emit logMessage("✅ FastSAM ONNX model loaded successfully.");
        return true;
    } catch (const cv::Exception& e) {
        emit logMessage(QString("❌ Exception loading ONNX: %1").arg(e.what()));
        return false;
    }
}

cv::Mat FastSamInference::removeBackground(const cv::Mat &inputImage)
{
    if (!_isModelLoaded || inputImage.empty()) {
        return inputImage;
    }

    try {
        cv::Mat blob;
        cv::dnn::blobFromImage(inputImage, blob, 1.0 / 255.0, cv::Size(_inputSize, _inputSize), cv::Scalar(), true, false);
        _net.setInput(blob);

        std::vector<cv::Mat> outputs;
        emit logMessage("⚡ Running FastSAM forward pass...");
        _net.forward(outputs, _net.getUnconnectedOutLayersNames());
        emit logMessage("⚡ FastSAM forward pass completed.");

        if (outputs.size() < 2) {
            emit logMessage("⚠️ FastSAM output size < 2, cannot process masks.");
            return inputImage;
        }
        
        cv::Mat outMask;
        processOutput(inputImage, outputs, outMask);
        
        if (outMask.empty()) {
            return inputImage; // Return original if no valid mask
        }
        
        // Apply mask to original image
        cv::Mat result;
        cv::bitwise_and(inputImage, inputImage, result, outMask);
        return result;
    } catch (const cv::Exception& e) {
        emit logMessage(QString("❌ cv::Exception in removeBackground: %1").arg(e.what()));
    } catch (const std::exception& e) {
        emit logMessage(QString("❌ std::exception in removeBackground: %1").arg(e.what()));
    } catch (...) {
        emit logMessage("❌ Unknown exception in removeBackground!");
    }
    return inputImage;
}

void FastSamInference::processOutput(const cv::Mat &inputImage, const std::vector<cv::Mat> &outputs, cv::Mat &outMask)
{
    if (outputs.size() < 2) return;
    
    emit logMessage(QString("FastSAM outputs: Out0 dims=%1, Out1 dims=%2")
                    .arg(outputs[0].dims).arg(outputs[1].dims));
                    
    cv::Mat boxes_mat, proto_mat;
    if (outputs[0].dims == 3 && outputs[0].size[1] == 37 && outputs[0].size[2] == 8400) {
        boxes_mat = outputs[0];
        proto_mat = outputs[1];
    } else if (outputs[1].dims == 3 && outputs[1].size[1] == 37 && outputs[1].size[2] == 8400) {
        boxes_mat = outputs[1];
        proto_mat = outputs[0];
    } else if (outputs[0].dims == 3 && outputs[0].size[2] == 37 && outputs[0].size[1] == 8400) {
        boxes_mat = outputs[0];
        proto_mat = outputs[1];
    } else if (outputs[0].dims == 2 && ((outputs[0].size[0] == 8400 && outputs[0].size[1] == 37) || (outputs[0].size[1] == 8400 && outputs[0].size[0] == 37))) {
        boxes_mat = outputs[0];
        proto_mat = outputs[1];
    } else if (outputs[1].dims == 2 && ((outputs[1].size[0] == 8400 && outputs[1].size[1] == 37) || (outputs[1].size[1] == 8400 && outputs[1].size[0] == 37))) {
        boxes_mat = outputs[1];
        proto_mat = outputs[0];
    } else {
        QString outStr = QString("❌ FastSAM Error: Unexpected model output dimensions! Out0: dims=%1, Out1: dims=%2").arg(outputs[0].dims).arg(outputs[1].dims);
        emit logMessage(outStr);
        return;
    }
    
    cv::Mat boxes_2d;
    if (boxes_mat.dims == 3) {
        if (boxes_mat.size[1] == 37 && boxes_mat.size[2] == 8400) {
            cv::Mat temp(boxes_mat.size[1], boxes_mat.size[2], CV_32F, boxes_mat.ptr<float>());
            boxes_2d = temp.t(); 
        } else {
            cv::Mat temp(boxes_mat.size[1], boxes_mat.size[2], CV_32F, boxes_mat.ptr<float>());
            boxes_2d = temp;
        }
    } else { // dims == 2
        if (boxes_mat.size[0] == 37 && boxes_mat.size[1] == 8400) {
            boxes_2d = boxes_mat.t();
        } else {
            boxes_2d = boxes_mat;
        }
    }
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> maskCoeffs;
    
    float rw = (float)inputImage.cols / _inputSize;
    float rh = (float)inputImage.rows / _inputSize;
    
    for (int i = 0; i < boxes_2d.rows; ++i) {
        float* row = boxes_2d.ptr<float>(i);
        float conf = row[4]; 
        if (conf > _confThreshold) {
            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];
            
            int left = int((cx - 0.5 * w) * rw);
            int top = int((cy - 0.5 * h) * rh);
            int width = int(w * rw);
            int height = int(h * rh);
            
            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(conf);
            classIds.push_back(0);
            
            std::vector<float> coeff(row + 5, row + 37);
            maskCoeffs.push_back(coeff);
        }
    }
    
    emit logMessage(QString("FastSAM: Found %1 boxes above confidence threshold").arg(boxes.size()));
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, _confThreshold, _iouThreshold, indices);
    
    if (indices.empty()) {
        emit logMessage("FastSAM: No boxes survived NMS. Returning empty mask.");
        return;
    }
    
    emit logMessage(QString("FastSAM: %1 boxes survived NMS.").arg(indices.size()));
    
    // Auto Center Prompting
    int centerX = inputImage.cols / 2;
    int centerY = inputImage.rows / 2;
    cv::Point centerPt(centerX, centerY);
    
    int bestIdx = -1;
    for (int idx : indices) {
        if (boxes[idx].contains(centerPt)) {
            bestIdx = idx;
            break; 
        }
    }
    
    if (bestIdx == -1) {
        bestIdx = indices[0];
    }
    
    cv::Mat p(32, 160 * 160, CV_32F, proto_mat.ptr<float>());
    cv::Mat c(1, 32, CV_32F, maskCoeffs[bestIdx].data());
    
    cv::Mat m = c * p; // [1, 25600]
    m = m.reshape(1, 160); // [160, 160]
    
    cv::Mat sig;
    cv::exp(-m, sig);
    sig = 1.0 / (1.0 + sig); // sigmoid
    
    cv::resize(sig, sig, cv::Size(inputImage.cols, inputImage.rows));
    cv::Mat mask = sig > 0.5; // Binary mask
    
    cv::Mat finalMask = cv::Mat::zeros(inputImage.size(), CV_8UC1);
    cv::Rect roi = boxes[bestIdx] & cv::Rect(0, 0, inputImage.cols, inputImage.rows);
    if (roi.area() > 0) {
        mask(roi).copyTo(finalMask(roi));
    }
    
    outMask = finalMask;
}
