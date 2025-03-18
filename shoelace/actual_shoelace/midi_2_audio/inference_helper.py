import torch
import torch.nn.functional as F
from shoelace.midi_lm.models.config import SEG_RES


class InferenceHelper:
    def __init__(self, model_folder, device):
        from shoelace.actual_shoelace.shoelace import Shoelace as Model
        from shoelace.actual_shoelace.midi_2_audio.config import MODEL_FACTORY
        self.model = Model(mask_type=None, device=torch.device("cuda"), model_configs=MODEL_FACTORY, model_names=["AudioLM", "MIDILM"])
        self.model.load_weights(model_folder)
        self.model.eval().to(device)

    
    @torch.no_grad()
    def inference(self, input_ids_generator, chunk_frame, hop_frame, top_k):
        audio_prompt = None
        n_id = 0
        results = []
        for input_ids, midi_index in input_ids_generator:

            if midi_index is None:
                break
            print(input_ids)
            print(midi_index)
            self.model.inference(model_name="MIDILM", max_len=(input_ids[0, :, 0] == SEG_RES).sum(),
                            use_generator=False, top_k=16, 
                            last_chunk=True, input_ids=input_ids, reset_cache=True)


            audio_codes = self.model.inference(model_name="AudioLM", 
                            cond_model_name="MIDILM", 
                            max_len=chunk_frame - hop_frame + 4 if n_id > 0 else chunk_frame + 4,
                            use_generator=True, top_k=top_k, reset_cache=True,
                            last_chunk=False, input_ids=audio_prompt, cond_indices=midi_index,
                            batch_size=len(input_ids), device=input_ids.device)
            results.append(audio_codes[:, :hop_frame])
            audio_prompt = audio_codes[:, hop_frame:chunk_frame]
            
            
            n_id += 1
            if n_id > 1:
                break
            
        results.append(audio_prompt)
        audio_codes = torch.concat(results, 1).transpose(1, 2)
        return audio_codes, input_ids



if __name__=="__main__":
    inference_helper = InferenceHelper()
    input_ids = torch.ones([1, 10, 6]).cuda().long()
    audio_codes = inference_helper.inference(input_ids, max_len=16*50)
    print(audio_codes.shape)
        
            