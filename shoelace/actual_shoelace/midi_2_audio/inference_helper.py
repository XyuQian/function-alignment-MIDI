import torch
from shoelace.utils.network_utils import transform_inputs
from shoelace.midi_lm.models.config import SEG_RES
import torch.nn.functional as F

class InferenceHelper:
    def __init__(self, model_folder):
        from shoelace.actual_shoelace.shoelace import Shoelace as Model
        from shoelace.actual_shoelace.midi_2_audio.config import MODEL_FACTORY
        self.model = Model(device=torch.device("cuda"), model_configs=MODEL_FACTORY, model_names=["AudioLM", "MIDILM"])
        self.model.load_weights(model_folder)
        self.model.cuda().eval()

    def inference(self, input_ids, max_len, top_k=100):
        
        midi_index = transform_inputs(input_ids[..., 0], SEG_RES).unsqueeze(-1).long()
        midi_index = F.pad(midi_index, (0, 0, 1, 0), "constant", 0)
        x = torch.arange(max_len + 5).unsqueeze(0).long().to(input_ids.device)
        audio_index = torch.stack([F.pad(x, (i + 1, 3 - i), "constant", 0) for i in range(4)], -1)


        midi_index = midi_index[..., 0]
        audio_index = x

        self.model.inference(model_name="MIDILM", max_len=1,
                            use_generator=False, top_k=16, 
                            last_chunk=True, input_ids=input_ids)


        audio_codes = self.model.inference(model_name="AudioLM", 
                            cond_model_name="MIDILM", max_len=max_len,
                            use_generator=True, top_k=top_k, 
                            last_chunk=True, input_ids=None, cond_indices=midi_index,
                            main_indices=audio_index,
                            batch_size=len(input_ids), device=input_ids.device)
        return audio_codes



if __name__=="__main__":
    inference_helper = InferenceHelper()
    input_ids = torch.ones([1, 10, 6]).cuda().long()
    audio_codes = inference_helper.inference(input_ids, max_len=16*50)
    print(audio_codes.shape)
        
            