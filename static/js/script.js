var errorCallback = function(e) {
console.log('Failed again!', e);
};

const WebCamElement = document.getElementById("webcam");
const WebCam = new Webcam(WebCamElement, "Camera Feed");
WebCam.flip()
WebCam.start();
