// var errorCallback = function(e) {
// console.log('Failed again!', e);
// };

// const WebCamElement = document.getElementById("webcam");
// const WebCam = new Webcam(WebCamElement, "Camera Feed");
// WebCam.flip()
// WebCam.start();
import * as tf from '@tensorflow/tfjs';
const model = await tf.loadLayersModel('file://wsl.localhost/Ubuntu/home/erwin/BE_Project/ISL_Gesture_Recognition/model.json');

const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
//   canvasCtx.drawImage(results.segmentationMask, 0, 0,
//                       canvasElement.width, canvasElement.height);

  // Only overwrite existing pixels.
  canvasCtx.globalCompositeOperation = 'source-in';
//   canvasCtx.fillStyle = '#00FF00';
  canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

  // Only overwrite missing pixels.
  canvasCtx.globalCompositeOperation = 'destination-atop';
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);

  canvasCtx.globalCompositeOperation = 'source-over';
//   drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
//                  {color: '#00FF00', lineWidth: 4});
//   drawLandmarks(canvasCtx, results.poseLandmarks,
//                 {color: '#FF0000', lineWidth: 2});
//   drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_TESSELATION,
//                  {color: '#C0C0C070', lineWidth: 1});
  drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
                 {color: '#CC0000', lineWidth: 3});
  drawLandmarks(canvasCtx, results.leftHandLandmarks,
                {color: '#13FF10', lineWidth: 0});
  drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
                 {color: '#00CC00', lineWidth: 3});
  drawLandmarks(canvasCtx, results.rightHandLandmarks,
                {color: '#FF0000', lineWidth: 0});
  canvasCtx.restore();
}

const holistic = new Holistic({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
}});
holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: true,
  smoothSegmentation: true,
  refineFaceLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
holistic.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await holistic.send({image: videoElement});
  },
  width: 360,
  height: 360
});
videoElement.style.visibility = "hidden";
videoElement.style.zIndex = "0";
canvasElement.style.zIndex = "1";

camera.start();
const example = tf.fromPixels(canvasElement);  // for example
const prediction = model.predict(example);
console.log(prediction)
