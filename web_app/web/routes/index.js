import predictionRoutes from './predictionRoutes.js';

const constructor = (app) => {
    app.use('/', predictionRoutes);

    app.use((req, res) => {
        res.status(404).json({
            error: 'Route Not Found'
        });
    });
};

export default constructor;