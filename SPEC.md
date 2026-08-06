# OCR/HTR Models Evaluation on IAM Database and Extended Datasets

## Project Overview

This project aims to evaluate and benchmark various Optical Character Recognition (OCR) and Handwriting Text Recognition (HTR) models on the IAM Handwriting Database and other complementary datasets. The objective is to systematically assess model performance, accuracy, speed, and suitability for different use cases.

## Objectives

1. **Model Evaluation**: Assess performance metrics of multiple state-of-the-art OCR/HTR models
2. **Comparative Analysis**: Compare models across various dimensions (accuracy, inference time, resource usage)
3. **Dataset Coverage**: Evaluate models on IAM database and extended datasets to ensure comprehensive assessment
4. **Benchmarking**: Establish baseline performance metrics for future improvements
5. **Documentation**: Provide detailed analysis and recommendations for model selection

## Datasets

### Primary Dataset
- **IAM Handwriting Database**
  - Description: Large-scale database of handwritten text
  - Coverage: Various writing styles and quality levels
  - Size: ~115,000 isolated handwritten words and ~50,000 text lines
  - Access: Requires registration at IAM website

### Secondary Datasets (To be determined)
- Additional publicly available handwriting or document datasets
- Potentially includes:
  - MNIST/EMNIST
  - RIMES dataset
  - CVL dataset
  - Custom domain-specific datasets (if applicable)

## Models to Evaluate

### OCR Models
- [ ] Tesseract
- [ ] Kraken
- [ ] DeepSeek OCR
- [ ] Chandra

### HTR Models
- [ ] PyLaia
- [ ] Qwen2-VL 8B (Vision-Language)

## Evaluation Metrics

### Accuracy Metrics
- Character Error Rate (CER): % of character-level errors
- Word Error Rate (WER): % of word-level errors
- Sequence Error Rate (SER): % of completely incorrect sequences
- Confidence scores (if applicable)

### Performance Metrics
- Inference time per image/sample
- GPU/CPU memory usage
- Throughput (samples per second)
- Model size (MB)

### Robustness Metrics
- Performance on degraded/low-quality images
- Performance on different writing styles
- Performance on different document types

## Methodology

### Pre-HPC Setup (Local/Interactive)
1. **Data Download**: Run `setup/download_data.sh` to fetch IAM and other datasets
2. **Model Preparation**: Run `setup/download_models.sh` to download/prepare all models
3. **Dependencies**: Install requirements via `setup/setup_environment.sh`
4. **Validation**: Test one model locally before HPC submission

### Data Preparation
1. Standardize image preprocessing across all models (in `evaluators/utils.py`)
2. Define train/validation/test splits
3. Document any augmentation techniques used
4. Ensure consistent evaluation protocol

### HPC Evaluation Process
1. Submit job via `hpc/submit_job.sh` to Adroit
2. SLURM job calls `hpc/run_evaluation.py` orchestrator
3. Orchestrator sequentially runs each model-specific evaluator
4. Each evaluator:
   - Loads pre-downloaded model from `models/` directory
   - Processes data from `data/` directory
   - Writes results to `results/` directory
   - Logs metrics and inference times
   - Documents any failures or edge cases

### Analysis & Reporting (Post-HPC)
1. Download results from Adroit
2. Compare results across models using Jupyter notebooks
3. Generate visualizations (charts, graphs)
4. Identify strengths and weaknesses of each model
5. Provide recommendations based on use case

## Project Structure

```
not_flawless/
├── SPEC.md (this file)
├── README.md
├── requirements.txt
├── setup/
│   ├── download_data.sh          # Download IAM and other datasets
│   ├── download_models.sh        # Download/prepare pre-trained models
│   └── setup_environment.sh      # Install dependencies
├── data/
│   ├── iam/                      # IAM database (downloaded)
│   ├── other_datasets/           # Additional datasets (downloaded)
│   └── processed/                # Preprocessed data (generated during job)
├── models/                       # Pre-downloaded models
│   ├── tesseract/
│   ├── paddleocr/
│   ├── easyocr/
│   └── [other_models]/
├── evaluators/                   # Model-specific evaluation scripts
│   ├── tesseract_eval.py
│   ├── paddleocr_eval.py
│   ├── easyocr_eval.py
│   ├── keras_ocr_eval.py
│   ├── trocr_eval.py
│   └── utils.py                  # Shared utility functions
├── hpc/
│   ├── submit_job.sh             # Main SLURM submission script
│   ├── run_evaluation.py          # Orchestrator for all evaluations
│   └── job_config.txt            # HPC parameters
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_results_analysis.ipynb
├── results/                      # Output directory (generated)
│   ├── metrics/
│   ├── visualizations/
│   ├── logs/
│   └── reports/
└── .gitignore
```

## Timeline & Milestones

- [ ] **Phase 1**: Finalize model list, dataset requirements
- [ ] **Phase 2**: Create setup scripts for data/model downloads
- [ ] **Phase 3**: Implement model-specific evaluator scripts
- [ ] **Phase 4**: Create HPC orchestrator and SLURM submission script
- [ ] **Phase 5**: Local testing and validation
- [ ] **Phase 6**: Submit and run HPC evaluation job
- [ ] **Phase 7**: Analysis and report generation

## Success Criteria

- [ ] All selected models successfully evaluated
- [ ] Comprehensive metrics collected for each model
- [ ] Clear ranking/comparison of models
- [ ] Documented findings and recommendations
- [ ] Reproducible evaluation pipeline

## Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Data access issues | Multiple dataset sources, pre-downloaded mirrors |
| Model dependency conflicts | Docker containers, virtual environments |
| Computational constraints | GPU access, model quantization if needed |
| Inconsistent evaluation | Standardized evaluation pipeline, automated testing |

## Future Work

- Fine-tune selected models on custom datasets
- Ensemble approaches
- Real-time inference optimization
- Integration with document processing pipelines
- Extended language support evaluation

## References

- IAM Database: https://fki.ics.unimaas.nl/databases/iam-handwriting-database/
- [Additional research papers and documentation TBD]

## Contact & Contributors

[To be populated]

---

# Della Churro — Semantic Error Experiment

Semantic Error in Churro experiment 

This is part of a larger project to identify a novel form of error when using VLMs for OCR/HTR. 

One experiment uses the IAM dataset to note that semantic errors occur even in data that is likely in a model's training data. 
https://github.com/apjanco/not_flawless

We're using this library to identify semantic errors.
https://github.com/apjanco/misnomer

In this experiment, we will be using the Churro dataset, with historical texts and a variety of languages.
https://huggingface.co/datasets/stanford-oval/churro-dataset

our main questions are:
Does fine-tuning improve semantic error rates? 
How does semantic error vary across languages? 
What other types of related errors appear in a historical dataset? 

The churro paper (https://arxiv.org/html/2509.19768v1) uses traditional WER and CER. How does semantic error challenge or confirm their findings? 

Compare semantic error rates in Churro 3b (https://huggingface.co/stanford-oval/churro-3B) against Qwen 2.5 VL (the base model used in the churro paper) on the test split of the churro dataset. Be sure to retain the main_language, languages, main_script and scripts columns in the output. 

Output example:
```python
{'image_path': 12044,
 'ground_truth': 'uͤber deſſen in Einhundert Reichsthaler Handgeld beſtehendes\nnachgelaſſenes Vermoͤgen nach Vorſchrift der Geſetze diſpo-\nnirt werden wird.\nVon dem Stadtgericht zu Egeln ſind des daſelbſt verſtor-\nbenen Toͤpfermeiſters, Jacob Chriſtian Zeiſigs, ſeit 1752\nund laͤnger abweſende, eigentlich aus Wittſtock gebuͤrtige,\nund nach ihrem Leben und Aufenthalt unbekannte Bruͤder,\nals der Schloͤſſer, Joachim Michael, und Huthmacher, Jo-\nhann Chriſtian Zeiſigs, oder auch derſelben Erben oder\nCeßionarien, edictaliter citirt, aus der von ihrem erſtge-\nnannten Bruder wegen ihrer 26 Rthlr, 8 Gr. 4 Pf. Erbe-\ngelder unterm 11ten Junii 1777 ihnen beſtellten, und auf\ndeſſelben nunmehro verkauften Hauſe noch eingetragen ſte-\nhenden Caution ihre Anſpruͤche geltend zu machen, widrigen-\nfalls ſi: in Termino peremtorio den 21ſten April 1790 fuͤr\ntodt erklaͤrt, ihre nachgelaſſene Erben praͤcludirt, und obge-\ndachte Caution im Hypothekenbuche wird geloͤſcht werden.\nWann Johann Ludwig Hartwig Schuͤtt, ein Sohn des\ngeweſenen und verſtorbenen Mitpaͤchters Schuͤtt hieſelbſt,\nvor einiger Zeit, ohne Leideserben zu hinterlaſſen, verſtor-\nhen; ſo werden alle und jede, welche an die Verlaſſenſchaft\ndeſſelben aus Erbrecht, wegen Schuld oder ſonſt aus irgend\neinem andern Grunde Anſprache haben oder zu haben ver-\nmeynen, hiedurch vorgeladen, ſich am 9ten September dieſes\nJahrs, Vormittags um 10 Uhr, vor dem hieſigen Gerichte\nperſoͤnlich oder durch hinlaͤnglich Bevollmaͤchtigte einzufinden,\nund ihre Anſprache ſodann anzugeben und rechtlicher Art nach\nzu bewahrheiten, unter dem Nachtheil, daß diejenigen,\nwelche dieſes nicht leiſten, mit ihren Forderungen und An-\nſpruͤchen ausgeſchloſſen und zum ewigen Stillſchweigen ver-\nwieſen werden ſollen.\nGroßen-Lukow, unweit Penzlin, am\n4ten Julius 1789.\nvon Holſteinſches Gericht hieſelbſt.\nAuf das cum eventuali oblatione ad cedendum bonis be-\ngleitete Geſuch des hieſigen Schutzjuden, Joſeph Milchel,\nwerden deſſen geſammte Glaͤubiger bey Strafe der Aus-\nſchließung, und ſeine Schuldner bey Strafe doppelter Zah-\nlung geladen, erſtere, ihre Forderungen an den Joſeph Muͤ-\nchel und ſein Vermoͤgen, und letztere, ihre Schuld an dem-\nſelben den 15ten September d. J. Vormittags 9 Uhr, vor\nhieſigem Herzogl. Stadtgerichte vollſtaͤndig und ohne Vorbe-\nhalt zu liquidiren, auch die Forderungen bey Strafe fuͤr bloße\nChirographarien geachtet zu werden, mit den in Haͤnden ha-\nbenden Original-Beweisthuͤmern zu belegen, und demnaͤchſt\ndie Bekanntmachung eines nahen Termins durch die Schwe-\nrinſchen Jntelligenzen allein zum Verſuch eines guͤtlichen\nAuskommens, unter dem Nachtheil, daß die Ausbleibenden\nan den Beſchluͤſſen der Einkommenden gebunden ſeyn ſollen,\nwie auch weitern Beſcheides gewaͤrtig zu ſeyn.\nSignatum\nMalchin, den 10ten Julii 1789.\nHerzogl. Stadtgericht.\nDa ich gewilliget bin, mein in der Hauptſtraße belegenes\nWohnhaus, nebſt Garten, Ländereyen, Pferde, Kühe, Acker-\ngeräth, zu verkaufen; es beſtehet ſolches in ein maßives Wohn-\nhaus, mit rothem Ziegel gedeckt, 54 Fuß breit, 60 Fuß tief,\nunten 4 Stuben 16 Fuß breit, 16 Fuß lang, 2 Kammern,\n8 Fuß lang, 8 Fuß breit, 2 Küchen, 2 Säle, 1 Vorder-\nSaal, 2 Bodens; eine 114 Fuß lang und 53 breite große Korn-\nScheune, von ſtarkem guten Holze gebauet; ein Wagenremiſe,\nalles im guten baulichen Stande, wobey ein großer Hof- und\nMiſtplatz; einen großen Garten hinter dem Hauſe, von\n6 Himpten Korn Einſaat, mit ſehr ſchönen Fruchtbäumen, ꝛc.\nbeſetzt; ein Garten zur Seite des Hauſes; 8 Morgen,\n7 Himpten Einſaat, ganz freyes Marſchland; 3 Morgen\nGeeſtland; eigen Torfmohr; 8 Kirchenſtühle und 2 Begräb-\nniſſe. Und können ſich Kaufliebhaber bey Unterſchriebenem,\nals Bewohner und Eigenthümer deſſelben, melden, die Con-\nditiones daſelbſt vernehmen, und ſoll nach annehmlichem Bot\nzugeſchlagen werden.\nRitzebüttel, den 25ſten Junii 1789.\nJoh. Fr. W. Neii, M. Dr.\nCorn. und Jan de Graaff, Gärtners zu Liſſe, zwiſchen Har-\nlem und Leiden in Holland, machen das geehrte Publicum be-\nkannt, daß ſie abliefern ſtarke und geſunde Hyacinten-Zwie-\nbeln, für Töpfe und Gläſer, bey Stücke und bey Hunderten,\ndas Stück von 2, 4, 6, 8, 10, 15 und 20 Fl. um in Par-\nterre zu pflanzen, von 5, 7, 10, 15, 20, 30, 40, 50, 60 und\n100 Fl. die 100 Stücke. Frühe und ſpäte Tulpen, von 3, 4,\n5, 10, 20, 30, 40 und 100 Fl. die 100 Stücke. Tros-Nar-\nciſſen, von 6, 8 und 10 Fl. die 100 Stücke. Jris von Per-\nſien, 6 Fl. die 100 Stücke. Doppelte Joncquilles, 10 Fl.\ndie 100 Stücke. Narcis von Zion, incomparable Orange\nPhönix, 2 ein halb Fl. die 100 Stücke. Crocus Corona Jmpe-\nrialis, Engliſche und Spaniſche Jris, Cirlamen, Jxia, Celien,\nMartagon, Auricula, Pruna-Vera, Pflanzblumen, Engliſche\nund Americaniſche Baum- und Heiſtergewächſen, Fruchtbäu-\nmen, als Pfirſchen, Birnen, Aepfeln, Kirſchen, Abricoſen, ꝛc. ꝛc.\nRoſenbäumen, in 10 Sorten, 15 Fl. die 100 Stücke; in 20 Sor-\nten, 20 Fl. die 100 Stücke; in 50 Sorten, 50 Fl. die 100 Stücke,\nin 100 Sorten, 100 Fl. die 100 Stücke; Centifoli-Roſen,\n5 Fl. die 100 Stücke; gelbe doppelte Roſen, 15 Fl. die\n100 Stücke. Aſpergi-Pflanzen, 30, 40 und 60 Fl. die\n100 Stücke. Getrocknete Gemüſe, weiße Bohnen, 3, 2 ein\nhalb, und 2 Fl. das Pfund. Zucker-Erbſen, 4, 3, 2 ein halb\nund 2 Fl. das Pfund. Grüne Erbſen, in Schaalen, 2 ein halb,\n2 und 1 ein halb Fl. das Pfund. Fitz- und Aſpergie-Bohnen,\n1 ein halb Fl. das Pfund. Antichocken, Blumenkraut, Kir-\nſchen, Aepfeln, Birnen, ꝛc. Gemüſe-Saamen, und alles,\nwas zu dergleichen Handlung gehöret.\nWir empfehlen uns die Gewogenheit von allen Kaufleuten\nund Liebhabern, und verſichern eine gute und aufrichtige Be-\ndienung, ꝛc.\nBey dem Commiſſair, Herrn Joh. Heinr. Hampe, in\nBraunſchweig, auf der Schoͤppenſtaͤdter-Straße wohnhaft,\niſt Creme de Bretagne blanc, der dem Mahagony- und allem\nuͤbrigen Holze einen feinen dem Pariſer Lacke gleichen Glanz\nmittheilet, das Pfund zu 18 Ggr. und das Viertelpfund zu\n4 Ggr. 6 Pf. Ferner Creme de Bretagne gris, der allem\nLeder, als Schuhe, Stiefeln, Kutſchen und Geſchirre, eine\nſchoͤne Schwaͤrze und Glanz giebt, und im mindeſten nicht\nabſchmutzt, das ganze Pfund zu 1 Rthlr. 4 Ggr. und das\nViertelpfund zu 7 Ggr. nebſt Gebrauchzettel gratis, in Com-\nmißion zu haben. Da beyde Theile ſeit einiger Zeit ſtarken\nAbſatz geſunden, und an weit entfernte Oerter geſandt, auch\nmit vielem Beyfall verbraucht worden ſind; ſo erbietet ſich\nder Verkaͤufer, wenn es nicht den angefuͤhrten Nutzen hat,\njedem, der es verlangt, ſein Geld zuruͤckzuzahlen. Jmgleichen\nder aͤchte Spaniſche Pommeranzen Extract, iſt noch allezeit\nbey mir zu haben in Glaͤſer à 4 einen halben Ggr.\nEs hat unterm 29ſten April dieſes Jahrs ein gewiſſer\nHerr Magiſter Schmidt, aus Neuwarth, an mir Endesbe-\nnannten wegen einer gewiſſen Angelegenheit geſchrieben,\nwelche ich ihn auch nach ſeinem Willen beſtens beſorgt habe;\naber da weder dem hieſigen noch dem Leipziger Poſtamte ein\nOrt desgleichen Namens bekannt iſt, ſo muß ich die Antwort\nſo lange an mir behalten, bis der Herr MagiſterSchmidt\nnoch einmal an mir ſchreiben, und dabey anzeigen wird, in\nwelcher Provinz der Ort Neuwarth liegt.\nJohann Samuel Feſecke,\naus Holle, in Sachſen.\nAm Donnerſtage, den 23ſten Julius, des Morgens um\n10 Uhr, ſoll in der Groͤningerſtraße, im Hauſe Nr. 37,\neine Parthey der allerbeſten Mallaga-Weine, ſo wie ſie aus\ndem Lande gekommen ſind, beſtehend in ganzen, halben und\nQuart-Booten, auch Faͤſſern von ungefaͤhr 18 Stuͤbchen,\nin oͤffentlicher Auction an den Meiſtbietenden verkauft wer-\nden, durch die Mackler Pubſt, Heins, Luͤbbers, Flohr,\nWunderlich, Engelhardt, Nienau, Pubſt, jun. Hoffmann,\nBuhrmann, Seipel, Linck, Fick, Gießer, Jarre, Lagers,\nLuͤders und Caro.\nDienſtag und Mittewochen, den 18ten und 19ten Auguſt,\nVormittags um 10 Uhr, ſoll auf dem Boͤrſenſaal die wohl-\nbekannte und auserleſene Neumannſche Gemaͤhlde-Samm-\nlung, von denen beruͤhmteſten und beſten Meiſtern, am Meiſt-\nbietenden verkauft werden, und zwar auf Ordre eines loͤbl.\nZehnpfenning-Amts, durch die Mackier Goverts und Lapor-\nterie, bey welchen der Catalogus hieruͤber abzufordern iſt.\nNB. Obige Sachen ſind Tages zuvor in beliebigem Augen-\nſchein zu nehmen.',
 'predicted_text': 'über dessen in Einhundert Reichsthalers Handgelb besichendes\nnachgelassene Vermögen nach Vorschrift des Gesetzes killos\nnicht werden weist.\n\nVom Landesgericht zu Tüsen und des Gefährts verfühen,\nherrn Löpernmeister, Jacob Christian Seifigs, seit 1732\nund länger aufbewahrt, eigentlich mit Wittinck gebräuchte,\nund nach ihren Leben und Aufenthalt unbekannte Orte,\nals der Schöldner, Joachim Michael und Hartmacher, Jo-\nhann Christian Zeihs, oder auch verliehene Erben der\nGeschäftsleute, etc., etc., aus der von ihrem erheb-\nnamenten Deuter wegen ihrer 26 Rikler, 8 Gr. 4 Pf. Erbs\ngebeten unter dem 21sten Janu 1777 ihnen beschieden, und auf\nbeschieden nummero verkaufte Haufe noch eingetragene\nStes\nhenden Caution ihre Ansprüche geltend zu machen, widrigsten\nsich in Termino peremtorio den 21sten April 1790 für\nnicht erfüllt, ihre nachgelassenen Erben verklagt, und\ndachte Caution im Hypothekenbuche wird gelöscht werden.\n\nMann Johann Ludwig Hartwig Schäfer, ein Sohn des\ngemeinsamen und verstorbenen Wittschafter Schäfer, hieselb\nvor einiger Zeit, ohne Leibeserben zu hinterlassen, verstor-\nben; so werden alle und jede, welche an die Wollenschaft\nbeschieden sind, und die Erben, wegen Schuldt oder sonst\nlautig gegen einen anderen Bruder Ansprüche haben oder zu haben\nwollen, bisdurch vorgeschlagen, sich am 1sten September dieses\nJahres, Vormittags um 10 Uhr, vor dem hiesigen Gerichte\npersönlich oder durch hundertjährige Bevollmächtigte einzuführen,\nund ihre Ansprüche johann anzugeben und rechtlicher Weise nach\nzu bewahrenden, unter dem Rüdberst, das diejenigen,\nwelche diese nicht wissen, mit ihren Forderungen und An-\nsprüchen aufgeschrieben und zum endigen Stichwesen ver-\nwiesen werden sollen. Großen Lufwer, umweit Penzlin, am\n4ten Juli 1789.\n\nvon Gollesinischen Gerichts beifüllt.\n\nAuf das vom eventuell oblatione ad cedendum contra\nder kleiste Verlust des hiesigen Schaububen, Joseph Wilhelm,\nwerden dessen gesamte Gläubiger den Erträge der Auf-\nschließung und seine Schuldiger den Erträge doppelter Auf-\nschließung geladen, seither ihre Forderungen an den Jofrich Wil-\nhelm und sein Vermögen, und letztere, ihre Schuldt an dem\nselben den 13ten September d. J. Vormittags 9 Uhr, vor\ndiesigem Gericht, Stadtgerichts vollständig und ohne Vorbe-\nhalt zu liquidieren, auch hiesiger Schuldner den Erträge die bloß-\nChirographien gesendet zu werden, mit den in Händen ha-\nbenden Original-Bewilligungen zu belegen, und kennlich\ndie Bestimmung eines nahen Lernings, auch die Schwe',
 'cer': 75.612473487854,
 'wer': 86.44067645072937,
 'inference_time': 4.29898202419281,
 'error': None,
 'semantic_error_count': 157,
 'obvious_error_count': 9,
 'semantic_has_error': True,
 'semantic_document_score': 0.6290080840365451,
 'semantic_document_error_type': 'partial',
 'semantic_document_embedding_similarity': 0.8136439323425291,
 'semantic_scorer_mode': 'full',
 'semantic_scorer_version': '1.0',
 'semantic_lm_model': 'Qwen/Qwen2.5-0.5B',
 'semantic_embedder_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
 'languages': ['German'],
 'main_language': 'German',
 'main_script': 'Latin',
 'scripts': ['Latin']}
```
This experiment is running on Princeton's Della cluster. https://researchcomputing.princeton.edu/systems/della
