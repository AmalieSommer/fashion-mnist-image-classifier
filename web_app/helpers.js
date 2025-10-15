//Helper functions...

export const getPredCategory = (predValue) => {
    let type;
    let predNumber = Number(predValue);

    //Based on Fashion MNIST categories return the pred value type based on the predicted number:
    switch (predNumber) {
        case 0:
            type = 'T-shirt/Top'
            break;
        case 1:
            type = 'Trouser';
            break;
        case 2:
            type = 'Pullover';
            break;
        case 3:
            type = 'Dress';
            break;
        case 4:
            type = 'Coat';
            break;
        case 5:
            type = 'Sandal';
            break;
        case 6:
            type = 'Shirt';
            break;
        case 7:
            type = 'Sneaker';
            break;
        case 8:
            type = 'Bag';
            break;
        case 9:
            type = 'Ankle boot';
            break;
        default:
            type = 'Failed to determine';
            break;
    }
    return type;
}