# Body-text diagnostic inputs

Published with the user's explicit authorization. This ZIP contains 63 PNG line
crops and draft transcriptions, including 47 corrections supplied during review.
Seven lines retain a needs-review decision. These are NOT approved gold labels.
No model weights or reviewer identities are included.

Source: [PiotrSty/impact-psnc-polish-ocr](https://huggingface.co/datasets/PiotrSty/impact-psnc-polish-ocr),
revision `c7cb156fb95d2880699c33725bbaf1fbc1008fea`, Polish IMPACT/PSNC ground truth.
Source metadata identifies the images as CC-BY-3.0:
https://creativecommons.org/licenses/by/3.0/
Changes: selected validation paragraph regions, converted to RGB PNG, cropped
into proposed lines, with user-supplied draft transcription corrections. Original
transcription strings remain in the manifest. Collection and page IDs preserve
links back to the pinned source metadata. No endorsement by source creators is implied.

SHA-256 of `body-dev-input.zip`:
`913ded2bd5d742099567a2652460baaf3c4a8bb16625492d0931607396dbdd55`

The Kaggle notebook downloads this ZIP automatically and checks its hash.
No manual Kaggle dataset upload is required. See
[`training/kaggle_body_dev_diagnostic.ipynb`](../../../training/kaggle_body_dev_diagnostic.ipynb).
