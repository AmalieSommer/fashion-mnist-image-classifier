import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from "url";
import configRoutes from './routes/index.js';
import exphbs from 'express-handlebars';


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();

app.use('/public', express.static(path.join(__dirname, 'public')));
app.use(express.json());
app.use(express.urlencoded({extended: true}));

app.engine('handlebars', exphbs.engine({defaultLayout: 'main'}));
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'handlebars');

configRoutes(app);

app.listen(3000, () => {
    console.log('Your routes will be running on http://localhost:3000');
})
