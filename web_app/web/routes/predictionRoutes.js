import multer from "multer";
import path from "path";
import { spawn } from "child_process";

import { Router } from "express";
import { title } from "process";
const router = Router();

const upload = multer({ dest: 'images/' }); //Setting destination folder for storing the uploaded files

router
.route('/')
.get(async (req, res) => {
    try {
        res.render('home', { title: 'Fashion MNIST Prediction Model' });
    } catch (e) {
        res.status(500).json({error: e});
    }
})
.post(upload.single('image'), async (req, res) => {

    try {
        const imgPath = path.join(__dirname, req.file.path);
        const child_process = spawn('python', ['predict.py', imgPath]);

        pred_result = '';

        child_process.stdout.on('data', (data) => {
            pred_result = data;
            console.log('Result from child-process: ' + data);
        });
        child_process.stderr.on('error', (err) => {
            console.error('Failed with: ' + err);
        });
        child_process.on('exit', (data) => {
            console.log('Python Script has exited!');
            res.json({ prediction: data });
        })
    } catch (error) {
        console.error(error);
        res.status(500).send("Error with uploading the file or making the prediction.");
    }
})

export default router;