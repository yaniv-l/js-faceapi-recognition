// Getting a reference the the file input
const imaageUpload = document.getElementById('imageUpload');
// Disabling the input button as faceapi is yet ready
imageUpload.disabled = true;
// Creating a span element to hold status notification
const label = document.createElement('span');
label.style.position = 'relative';
label.innerHTML = "Loading faceapi...";
document.body.append(label);

// Loading all faceapi required models
Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')]).then(start);


async function start() {
    // Creating a div container to hold the image and elements
    const container = document.createElement('div');
    container.style.position = 'relative';
    document.body.append(container);
    // Loading all available face detections
    const labeledFaceDescriptors = await loadLabeledImages()
    // Creating a face matcher to be used later - face matcher will used learend face detections and will match when matching is 60% or more
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
    let image
    let canvas
    // All faceapi are ready now
    label.innerHTML = 'Ready... choose your image';
    // Enabling the file input button
    imageUpload.disabled = false;
    // Addging an event listener for the file input on file (image) was uploaded.
    imaageUpload.addEventListener('change', async () => {
        // If we already have an image and canvas - clears them
        if (image) image.remove()
        if (canvas) canvas.remove()
        // Conver the buffer into image and append it to the container
        image = await faceapi.bufferToImage(imaageUpload.files[0]);
        container.append(image);
        // Create the canvas on which face boxs will be drawn based on the the image
        canvas = faceapi.createCanvasFromMedia(image);
        container.append(canvas);
        // Resize the canvas to match the dimention of the imnge
        const displaySize = { width: image.width, height: image.height };
        faceapi.matchDimensions(canvas, displaySize);
        // Detecting all images in the current loaded image - with face landmarks and descriptors
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
        // Resize the reults based on display size
        const resizeDetection = faceapi.resizeResults(detections, displaySize);
        // Trying to match a lable for each face detected
        const results = resizeDetection.map(d => faceMatcher.findBestMatch(d.descriptor))
        // Looping all over the results and dawing the face box with matching label
        results.forEach((result, i) => {
            const box = resizeDetection[i].detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
            drawBox.draw(canvas);
        });

        // Number of faces detected in the image
        document.body.append(detections.length);
    })
}

function loadLabeledImages() {
    // Create a label for each face
    const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark']
    return Promise.all(
        labels.map(async label => {

            const descriptions = []
            // Each folder has 2 images for this face detection model
            for (let i = 1; i <= 2; i++) {
                const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/yaniv-l/js-faceapi-recognition/master/labeled_images/${label}/${i}.jpg`)
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }
            // 'dictionary element of the label and its face detection models'
            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}
