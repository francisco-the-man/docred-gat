def add_args(parser):
    # Data args
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--transformer_type", type=str, default="bert")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--train_file", type=str, default="train_annotated.json")
    parser.add_argument("--dev_file", type=str, default="dev.json")
    parser.add_argument("--test_file", type=str, default="test.json")
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--load_path", type=str, default="")
    parser.add_argument("--pred_file", type=str, default="pred.json")
    
    # Training args
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--tokenizer_name", type=str, default="")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--num_class", type=int, default=97)
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--max_sent_num", type=int, default=25)
    parser.add_argument("--evi_thresh", type=float, default=0.2)
    
    # Model args
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_transformer", type=float, default=5e-5)
    parser.add_argument("--lr_added", type=float, default=1e-4)
    
    # Other args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--display_name", type=str, default="")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--eval_mode", type=str, default="normal")
    parser.add_argument("--evaluation_steps", type=int, default=2500)
    parser.add_argument("--save_attn", action="store_true")
    parser.add_argument("--teacher_sig_path", type=str, default="")
    parser.add_argument("--evi_lambda", type=float, default=0.1)
    parser.add_argument("--attn_lambda", type=float, default=0.1)
    
    return parser 