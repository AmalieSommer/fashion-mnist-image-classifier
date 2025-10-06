const input = document.getElementById('img');
input.addEventListener('change', () => {
    const img = input.files;
    if(img) {
        const imgReader = new FileReader();
        const preview = document.getElementById('img-preview');

        imgReader.onload((event) => {
            preview.setAttribute('src', event.target.result);
        });
        imgReader.readAsDataURL(img[0]);
    } 
});