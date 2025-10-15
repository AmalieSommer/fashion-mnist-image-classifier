import multer from "multer";
import path from "path";
import { spawn } from "child_process";
import { Router } from "express";
import { fileURLToPath } from "url";
import { getPredCategory, getModelType } from '../helpers.js';


const router = Router();
const upload = multer({ dest: 'images/' }); //Setting destination folder for storing the uploaded files


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.join(__dirname, '..', '..'); //Getting the path to the root directory.
const predScript = path.join(__dirname, '../../model_scripts/predict.py');

router
.route('/')
.get(async (req, res) => {
    try {
        res.render('home', { title: 'Fashion MNIST Prediction Model' });
    } catch (e) {
        console.error('An error occurred with rendering home page!!!');
        res.status(500).json({error: e});
    }
})
.post(upload.single('image'), async (req, res) => {

    try {
        const imgPath = path.join(rootDir, req.file.path);
        const modelChoice = req.body.model;

        const child_process = spawn('python', [predScript, '--model', modelChoice, '--image', imgPath]);

        let pred_result = '';

        child_process.stdout.on('data', (data) => {
            pred_result += data;
            //console.log('Result from child-process: ' + data);
        });

        child_process.stderr.on('data', (err) => {
            console.error('Failed with: ' + err);
        });
        child_process.on('error', (err) => {
            console.error('Failed to start Python process:', err);
        });

        child_process.on('close', (code) => {
            const predCategory = getPredCategory(pred_result.trim());

            res.json({ 
                result: predCategory,
                chosenModel: getModelType(modelChoice)
             })
        })
    } catch (error) {
        console.error(error);
        res.status(500).send("Error with uploading the file or making the prediction.");
    }
})

export default router;