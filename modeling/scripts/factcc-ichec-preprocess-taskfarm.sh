# Run each in a session, then run taskfarm
# test_sentence_reference_resolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/test_sentence_reference_resolved \
--coref

# test_sentence_reference_unresolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/test_sentence_reference_unresolved

# test_sentence_decoded_resolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/test_sentence_decoded_resolved \
--coref

# test_sentence_decoded_unresolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/test_sentence_decoded_unresolved

# test_paragraph_reference_resolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/test_paragraph_reference_resolved \
--coref \
--method paragraph

# test_paragraph_reference_unresolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/test_paragraph_reference_unresolved \
--method paragraph

# test_paragraph_decoded_resolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/test_paragraph_decoded_resolved \
--coref \
--method paragraph

# test_paragraph_decoded_unresolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/test_paragraph_decoded_unresolved \
--method paragraph

# val_sentence_reference_resolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/val_sentence_reference_resolved \
--coref

# val_sentence_reference_unresolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/val_sentence_reference_unresolved

# val_sentence_decoded_resolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/val_sentence_decoded_resolved \
--coref

# val_sentence_decoded_unresolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/val_sentence_decoded_unresolved

# val_paragraph_reference_resolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/val_paragraph_reference_resolved \
--coref \
--method paragraph

# val_paragraph_reference_unresolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/val_paragraph_reference_unresolved \
--method paragraph

# val_paragraph_decoded_resolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/val_paragraph_decoded_resolved \
--coref \
--method paragraph

# val_paragraph_decoded_unresolved
python modeling/score.py \
--mode preprocess \
--cnndm $PWD/../data/cnndm \
--summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded \
--evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/val_paragraph_decoded_unresolved \
--method paragraph
