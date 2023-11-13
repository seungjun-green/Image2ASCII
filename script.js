document.getElementById('imageUploader').addEventListener('change', function(event) {
    var output = document.getElementById('preview');
    output.src = URL.createObjectURL(event.target.files[0]);
    output.onload = function() {
        URL.revokeObjectURL(output.src) // Free memory
        output.style.display = 'block'; // Show the image
    }
});


window.onload = function() {
    document.getElementById('displayArea').innerText = "Generated ASCII Art will be shown here.";
    document.getElementById('displayArea').style.fontSize = "large";
}

document.getElementById('convertToTensor').addEventListener('click', function() {


    // Ensure that TensorFlow.js is loaded
    if (!tf) {
        console.error('TensorFlow.js is not loaded!');
        return;
    }
    
    // Get the image element
    let imageElement = document.getElementById('preview');

    // Make sure an image is uploaded
    if (!imageElement.src || imageElement.style.display === 'none') {
        alert('Please upload an image first. :)');
        return;
    }

    document.getElementById('displayArea').innerText = "Generating....";
    document.getElementById('displayArea').style.fontSize = "large";
    
    setTimeout(() => {
    // STEP1. Get image tensor from uploaded image: (width, height, 3)
    imageTensor = getTensorFromImage(imageElement);

    // STEP2. Convert image into gray scale: (width, height, 1)
    gray_tensor = rgbToGrayscale(imageTensor);


    const iterations=5;
    const width = gray_tensor.shape[0];
    const height = gray_tensor.shape[1];
    const num_of_elements = width * height;
    
    let gray_tensor_1D = tf.reshape(gray_tensor, [num_of_elements])
    let gray_tensor_2D = tf.stack(Array(8).fill(gray_tensor_1D), 1);
    
    let centroid_dict = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]};
    let centroid_dict_res = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]};

    // Only for the first time, randomaly initallize location of each centroid.
    let centroids_1D = tf.randomUniform([8], 0, 256); 

    // STEP3
    // REPEAT:
    //        assign centroid for all elements
    //        update location of centroids
    for (let iteration = 0; iteration < iterations; iteration++) {


        //Step3-1: assign centroid for all elements(widt*hieght number of elements)
        // get distance vector shape of (num_of_elements, 8) and get indicesOfMinValues array from
        // it by selecting the column index which has the smallest value.
        // as a result we're assigining centroid for all elements
        let centroids_2D = tf.stack(Array(num_of_elements).fill(centroids_1D), 0);
        let difference = tf.sub(gray_tensor_2D, centroids_2D);
        let distance = tf.square(difference);
        let indicesOfMinValues = Array.from(tf.argMin(distance, 1).dataSync());  
        
        
        // STEP3-2: update location of centroids
        // based on current centroids, update the each centroid's location as mean of its elements
        // Update centroid_dict and centroid_dict_res dict.
        // centroid_dict; each value: 0 ~ 256
        // centroid_dict_res; each value: 0 ~ num_of_elements
        for (let i = 0; i < num_of_elements; i++) {
            centroid_dict[indicesOfMinValues[i]].push(gray_tensor_1D[i]);
            centroid_dict_res[indicesOfMinValues[i]].push(i);
        }
        
        let centroids_1D_new = [0, 0, 0, 0, 0, 0, 0, 0];
        
        // create an array containing new location of each centroid
        for (let key in centroid_dict) {
            if (centroid_dict[key].length > 0) {
                centroids_1D_new[key] = tf.mean(centroid_dict[key])
            }  
        }
        
        // update centroid's locations.
        centroids_1D = centroids_1D_new;
    }
    

    // STEP6. Construcy output_string from result array.
    final_string = constructFinalString(centroid_dict_res, width, height);

    document.getElementById('displayArea').innerText = final_string;
    if (window.innerWidth <= 300) {
        document.getElementById('displayArea').style.fontSize = '1%';
    } else if (window.innerWidth <= 768) {
        document.getElementById('displayArea').style.fontSize = '7px';
    } else {
        // Set font size for non-mobile devices
        document.getElementById('displayArea').style.fontSize = "x-small";
    }
    

    }, 0)
    
            
});


function constructFinalString(centroid_dict_res, width, height) {
    let result = new Array(width * height).fill(0);
    
    Object.keys(centroid_dict_res).forEach(key => {
        centroid_dict_res[key].forEach(element => {
            result[element] = parseInt(key); 
        });
    });
    // now each element in result array shows what centroid it belongs to    
        
    const charWeights = {
        0: '-',
        1: '=',
        2: ':',
        3: '+',
        4: '*',
        5: '%',
        6: '#',
        7: '@'
    };
    
    
    let final_string = ""
    
    // creating multi-line string(100 num of characters) * multi-lines
    for (let i = 0; i < result.length; i++) {
        final_string += charWeights[result[i]];
        if ((i+1)%100 == 0) {
            final_string+='\n';
        }
    }

    return final_string
}


function reshapeTensor2(tensor) {
    const height = tensor.shape[1];
    const width = tensor.shape[0];
    const reshapedTensor =  tensor.reshape([width, height]);

    return reshapedTensor
};

function rgbToGrayscale(imageTensor) {
    const [red, green, blue] = tf.split(imageTensor, 3, 2);

    const R_WEIGHT = 0.299;
    const G_WEIGHT = 0.587;
    const B_WEIGHT = 0.114;

    // Compute the grayscale values
    const grayscale = red.mul(R_WEIGHT).add(green.mul(G_WEIGHT)).add(blue.mul(B_WEIGHT))
    
    return reshapeTensor2(grayscale);
};


function getTensorFromImage(imageElement) {
    // Calculate new dimensions
    let originalWidth = imageElement.naturalWidth;
    let originalHeight = imageElement.naturalHeight;
    let newWidth = 100;
    let newHeight = Math.round((originalHeight / originalWidth) * 54.4);

    // Create a canvas element to perform the resize
    let canvas = document.createElement('canvas');
    canvas.width = newWidth;
    canvas.height = newHeight;
    let ctx = canvas.getContext('2d');

    // Draw the image resized on the canvas
    ctx.drawImage(imageElement, 0, 0, newWidth, newHeight);

    // Convert the canvas to a tensor
    let tensor = tf.browser.fromPixels(canvas).toFloat()
    reshapedTensor = tensor.reshape([tensor.shape[1], tensor.shape[0], 3]);

    return reshapedTensor
}