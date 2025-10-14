
$(document).ready(() => {

    //Read uploaded file:
    $('#image').on('change', (event) => {

        const img = event.target.files[0];
        if(img) {
            let fileReader = new FileReader();
            fileReader.onload = e => {
                $('#filePreview').attr("src", e.target.result);
            };
            fileReader.readAsDataURL(img);
        }
    })


    //Handle file upload submissions:
    $('#fileUpload').on('submit', (event) => {
        event.preventDefault();
        console.log('Submission form button clicked!');

        //Check if an image file has been uploaded:
        let file = $('#image')[0].files[0];

        console.log('Read file: ', file);

        if(file) {
            console.log('A file has been uploaded successfully!');
        } else {
            console.error('File is missing!');
        }


        //Send the uploaded image to the correct POST route using Ajax:
        const data = new FormData($('#fileUpload')[0]);
        console.log('New formdata: ', data);

        $.ajax({
            url: '/',
            type: 'POST',
            data: data,
            contentType: false,
            processData: false,
            success: (res) => {
                console.log('Upload success: ', res);
                $('#predCategory').text(res.result);
                $('#divResult').show();
            },
            error: (err) => {
                console.error('Upload failed: ', err);
            }
        });
    });
    
});