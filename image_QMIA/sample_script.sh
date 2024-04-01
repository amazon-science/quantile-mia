#Single run full example (no HPO)
MODEL_DIR=./models/
DATA_DIR=./data/
DATASET=cinic10/0_16
BASE_ARCHICECTURE=cifar-resnet-50
QMIA_ARCHITECTURE=facebook/convnext-large-224-22k-1k
# Train base model
python train_base.py --dataset=$DATASET --scheduler=step --scheduler_step_gamma=0.2 --scheduler_step_fraction=0.3 --lr=0.1 --weight_decay=5e-4 --batch_size=32 --image_size=32 --architecture=$BASE_ARCHITECTURE --epochs=200 --model_name_prefix=example --model_root=$MODEL_DIR --data_root=$DATA_DIR
# Train QMIA (No HPO)
python train_mia.py         --dataset=$DATASET  --epochs=10   --batch_size=16  --image_size=224 --use_hinge_score=True --use_target_label=True --model_name_prefix=gaussian_qmia --base_model_name_prefix=example   --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE  --tune_batch_size=True --use_gaussian=True --use_target_inputs=True --model_root=$MODEL_DIR --data_root=$DATA_DIR
# Evaluate performance
python plot_results.py         --dataset=$DATASET  --epochs=10   --batch_size=16  --image_size=224 --use_hinge_score=True --use_target_label=True --model_name_prefix=gaussian_qmia --base_model_name_prefix=example   --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE  --tune_batch_size=True --use_gaussian=True --use_target_inputs=True --model_root=$MODEL_DIR --data_root=$DATA_DIR



#Single run full example (HPO)
# Train base model
python train_base.py --dataset=$DATASET --scheduler=step --scheduler_step_gamma=0.2 --scheduler_step_fraction=0.3 --lr=0.1 --weight_decay=5e-4 --batch_size=32 --image_size=32 --architecture=$BASE_ARCHITECTURE --epochs=200 --model_name_prefix=example --model_root=$MODEL_DIR --data_root=$DATA_DIR
# Train QMIA (HPO)
python train_mia_ray.py         --dataset=$DATASET  --num_tune_samples=16 --epochs=6   --batch_size=16  --image_size=224 --use_hinge_score=True --use_target_label=True --model_name_prefix=gaussian_qmia_hpo --base_model_name_prefix=example   --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE  --tune_batch_size=True --use_gaussian=True --use_target_inputs=True --model_root=$MODEL_DIR --data_root=$DATA_DIR
# Evaluate performance
python plot_results.py         --dataset=$DATASET  --epochs=6   --batch_size=16  --image_size=224 --use_hinge_score=True --use_target_label=True --model_name_prefix=gaussian_qmia_hpo --base_model_name_prefix=example   --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE  --tune_batch_size=True --use_gaussian=True --use_target_inputs=True --model_root=$MODEL_DIR --data_root=$DATA_DIR
