

function testing() {
    console.log("Yout button worked")
}

document.getElementById('imageUploader').addEventListener('change', function(event) {
    var output = document.getElementById('preview');
    output.src = URL.createObjectURL(event.target.files[0]);
    output.onload = function() {
        URL.revokeObjectURL(output.src) // Free memory
        output.style.display = 'block'; // Show the image
    }
});



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
        alert('Please upload an image first.');
        return;
    }

    
    
    
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
       

    reshapedTensor = reshapeTensor(tensor);

    gray_tensor = rgbToGrayscale(reshapedTensor);


    let centroids_1D = tf.randomUniform([8], 0, 256); 


    const iterations=5;
    const width = gray_tensor.shape[0];
    const height = gray_tensor.shape[1];
    const num_of_elements = width * height;
    
    let gray_tensor_1D = tf.reshape(gray_tensor, [num_of_elements])
    let gray_tensor_2D = tf.stack(Array(8).fill(gray_tensor_1D), 1);
    
    let centroid_dict = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]};
    let centroid_dict_res = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]};

    for (let iteration = 0; iteration < 2; iteration++) {
        let centroids_2D = tf.stack(Array(num_of_elements).fill(centroids_1D), 0);
        let difference = tf.sub(gray_tensor_2D, centroids_2D);
        let distance = tf.square(difference);
        let indicesOfMinValues = Array.from(tf.argMin(distance, 1).dataSync());
        
        console.log(indicesOfMinValues);
        

        for (let i = 0; i < num_of_elements; i++) {
            centroid_dict[indicesOfMinValues[i]].push(gray_tensor_1D[i]);
            centroid_dict_res[indicesOfMinValues[i]].push(i);
            
        }
        
        let centroids_1D_new = [0, 0, 0, 0, 0, 0, 0, 0];
        
        for (let key in centroid_dict) {
            console.log(centroid_dict[key].length);
            if (centroid_dict[key].length > 0) {
                 centroids_1D_new[key] = tf.mean(centroid_dict[key])
            }  
        }
        
        centroids_1D = centroids_1D_new
    }
    
    
    let result = new Array(width * height).fill(0);
        

    Object.keys(centroid_dict_res).forEach(key => {
        centroid_dict_res[key].forEach(element => {
            result[element] = parseInt(key); 
        });
    });
            
        
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
    
    for (let i = 0; i < result.length; i++) {
        final_string += charWeights[result[i]];
        if ((i+1)%100 == 0) {
            final_string+='\n';
        }
    }
        

    console.log(final_string);
        
    document.getElementById('displayArea').innerText = final_string;
            
});


function reshapeTensor(tensor) {
    const height = tensor.shape[0];
    const width = tensor.shape[1];
    const reshapedTensor =  tensor.reshape([width, height, 3]);

        return reshapedTensor
};

function reshapeTensor2(tensor) {
    const height = tensor.shape[1];
    const width = tensor.shape[0];
    const reshapedTensor =  tensor.reshape([width, height]);

    return reshapedTensor
};

function rgbToGrayscale(imageTensor, height, width) {
    console.log(imageTensor.shape)
    const [red, green, blue] = tf.split(imageTensor, 3, 2);
    console.log(red.shape, green.shape, blue.shape);

    const R_WEIGHT = 0.299;
    const G_WEIGHT = 0.587;
    const B_WEIGHT = 0.114;

    // Compute the grayscale values
    const grayscale = red.mul(R_WEIGHT)
                         .add(green.mul(G_WEIGHT))
                         .add(blue.mul(B_WEIGHT))
//                        
    
    return reshapeTensor2(grayscale);
};




