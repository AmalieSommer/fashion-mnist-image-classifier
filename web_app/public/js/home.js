
$(document).ready(() => {

    //Read uploaded file:
    $('#image').on('change', (event) => {

        const img = event.target.files[0];
        if(img) {
            let fileReader = new FileReader();
            fileReader.onload = e => {
                $('#filePreview').attr("src", e.target.result);
                $('#filePreview').show();
            };
            fileReader.readAsDataURL(img);
        }
    })


    //Handle file upload submissions:
    $('#fileUpload').on('submit', (event) => {
        event.preventDefault();

        //Check if an image file has been uploaded:
        let file = $('#image')[0].files[0];

        if(file) {
            console.log('A file has been uploaded successfully!');
        } else {
            console.error('File is missing!');
        }


        //Send the uploaded image to the correct POST route using Ajax:
        const data = new FormData($('#fileUpload')[0]);
        const selected_model = $('#ml-model-select').find(':selected').val();
        data.append('model', selected_model); //Creates formdata with file and selected option attached.

        $.ajax({
            url: '/',
            type: 'POST',
            data: data,
            contentType: false,
            processData: false,
            success: (res) => {
                console.log('Upload success: ', res);
                $('#predCategory').text(res.result);
                $('#modelSelection').text(res.chosenModel);
                $('#divResult').show();

                $('#fileUpload')[0].reset();
                $('#filePreview').hide();
            },
            error: (err) => {
                console.error('Upload failed: ', err);
            }
        });
    });
    
});