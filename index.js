const NodeWebcam = require('node-webcam')
const fs = require('fs')
const moment = require('moment')
const tf = require('@tensorflow/tfjs')
const tfnode = require('@tensorflow/tfjs-node')
const faceapi = require('@vladmandic/face-api')
const { loadImage, Canvas, Image, ImageData } = require('canvas')
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })

// Initializing folder & camera
let cameraOptions = { width: 1280, height: 720, quality: 100, frames: 60, delay: 0, saveShots: true, output: 'png', callbackReturn: 'base64', device: false, verbose: false }
let Webcam = NodeWebcam.create(cameraOptions)
if (!fs.existsSync('frames')) {
  fs.mkdirSync('frames')
}

// Initializing face-models
let FACE_MODEL = './models/'
faceapi.nets.ssdMobilenetv1.loadFromDisk(FACE_MODEL)
.then(faceapi.nets.tinyFaceDetector.loadFromDisk(FACE_MODEL))
.then(faceapi.nets.faceLandmark68Net.loadFromDisk(FACE_MODEL))
.then(faceapi.nets.faceLandmark68TinyNet.loadFromDisk(FACE_MODEL))
.then(faceapi.nets.faceRecognitionNet.loadFromDisk(FACE_MODEL))
.catch(error => { console.log(error) })

// Main function
function main() {
  Webcam.capture('frames/frame', async function (error,) {
    const loadImageCanvas = await loadImage('./frames/frame.png')
    const faceDetections = await faceapi.detectSingleFace(loadImageCanvas).withFaceLandmarks(false).withFaceDescriptor()
    if (faceDetections) {
      const findFace = JSON.parse(fs.readFileSync(`./data/me.json`, 'utf8'))
      const faceData = new faceapi.FaceMatcher(findFace.map(x => faceapi.LabeledFaceDescriptors.fromJSON(x)), 0.5)
      const faceMatch = faceData.findBestMatch(faceDetections.descriptor)

      if (faceMatch.label !== 'unknown') {
        console.log(`[${moment().format('HH:mm:ss')}] Face is found : ${faceMatch.label}`)
      } else {
        console.log(`[${moment().format('HH:mm:ss')}] Face not found`)
      }
    }
    Webcam.clear()
    main()
  })
}

main()