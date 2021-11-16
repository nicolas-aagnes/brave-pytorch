# BraVe Pytorch

This is a pytorch implementation of the Broaden Your Views paper from DeepMind.

BraVe codestructure:
- Each module will have its own cli arguments.
- Smart to have an experiment wide config that has module configs inside it. So each model has its own respective config.
- 

CLI Solution:
1. Each module is a pl module
2. Could first use parser.parse_known_args to get the names of the teacher and student models.
3. Add the arguments of the respective modules to parser
4. Use parser.get_args to get all args for all modules.
5. Each core model needs a script for creating a model_config.yaml file.
6. Example usage:

python models/tsm_config.py --num-features 64 --timesteps 10  tmp/student.json
python models/tsm_config.py --num-features 64 --timesteps 10  tmp/teacher.json

python train.py \
    --batch-size 128
    --student-config tmp/student.json \
    --teacher-config tmp/teacher.json

rm -rf tmp

Questions
1. Should the student and teacher modules be nn.Module or pl.LightningModule? Should test how logging, hparams, optimizers work in different scenarios. https://github.com/PyTorchLightning/pytorch-lightning/discussions/7249

For an ensamble model: (used this for my analysis https://github.com/PyTorchLightning/pytorch-lightning/discussions/7249)
- hparams can be passed around nicely
- 