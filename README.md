# huggingface-stt-transformer-lm-integration

### Features
- used batch inference to speedup beam scoring (unlike the original fairseq c++ implementation)
- applied beam pruning by selecting topk vocabs to further speedup
- can be used with any kind of causal lm models (GPT-Neo, GPT2, ...)

### usage
This code can be applied for any acoustic model that uses [CTC Algorithm](https://distill.pub/2017/ctc/) for its output, such as Wav2vec 2.0.
```python
class TransformerDecoder:
    def __init__(self, 
                 processor, 
                 gpt, 
                 gpt_tokenizer, 
                 beam_width=50, 
                 num_topk=10, 
                 alpha=0.5, 
                 beta=1.0, 
                 score_interval=50):
        """
        processor (`Wav2Vec2Processor`):
            An instance of Wav2Vec2Processor
        gpt (`GPT2LMHeadModel`):
            An instance of GPT2LMHeadModel. GPT2 (or any LM model) is used for scoring beams.
        gpt_tokenizer (`GPT2Tokenizer`):
            An instance of GPT2Tokenizer corresponding to `gpt`.
        beam_width (`int`, defaults to 50):
            Beam width to do beam search with.
        num_topk (`int`, defaults to 10):
            Number of words to select from. This is one of the beam pruning parameter to optimize GPU memory and inference time.
        alpha (`int`, defaults to 0.5):
            Parameter to fusion logits with LM score. Beam score is determined as `(score from acoustic model) + alpha * (score from LM) + beta`
        beta (`int`, defaults to 1.0):
            Parameter to fusion logits with LM score. Beam score is determined as `(score from acoustic model) + alpha * (score from LM) + beta`
        score_interval (`int`, defaults to 50):
            Interval to score every beam. Interval in seconds is (20 ms * score_interval)
        """
        ...

    def decode(self, logits):
        """
        logits (`numpy.ndarray`):
            Output logits of Wav2vec2 model with shape (output_length, num_vocab) as numpy ndarray.
        """
        ...
```

Some results with `elgeish/wav2vec2-base-timit-asr` and `gpt2` on [TIMIT asr dataset](https://huggingface.co/datasets/timit_asr):
```text
label   :	 she had your dark suit in greasy wash water all year 
pred_raw:	 she had your dark suit in greasy wash water all year
pred_gpt:	 she had your dark suit in greasy wash water all year
-----------
label   :	 she had your dark suit in greasy wash water all year 
pred_raw:	 she had your dark suit in greasy wash water all year
pred_gpt:	 she had your dark suit in greasy wash water all year
-----------
label   :	 there are more obvious nymphomaniacs on any privateeye series 
pred_raw:	 ther are more obvious nimpfom many acts on any privite eye series
pred_gpt:	 there are more obvious nimfom many acts on any private eye series
-----------
label   :	 only the best players enjoy popularity 
pred_raw:	 only the best players enjoy popularity
pred_gpt:	 only the best players enjoy popularity
-----------
label   :	 december and january are nice months to spend in miami 
pred_raw:	 tdisember and january ar nice months toespend in my amy
pred_gpt:	 disember and january are nice months to spend in my amy
-----------
label   :	 keep the thermometer under your tongue 
pred_raw:	 keep the thermometer under your tongue
pred_gpt:	 keep the thermometer under your tongue
-----------
label   :	 you're a taxpayer householder landlord 
pred_raw:	 you'r a taxpayer householder landlord
pred_gpt:	 you're a taxpayer householder landlord
-----------
label   :	 does creole cooking use curry 
pred_raw:	 does creole coking use curry
pred_gpt:	 does creole cooking use curry
-----------
label   :	 scholastic aptitude is judged by standardized tests 
pred_raw:	 scholastic aptitude is judged by standardized tests
pred_gpt:	 scholastic aptitude is judged by standardized tests
-----------
label   :	 are you looking for employment 
pred_raw:	 are oloking for iemployment
pred_gpt:	 are looking for employment
```