import torch
import torch.nn.functional as F
from shoelace.midi_lm.models.config import SEG_RES, RES_EVENT

def cut_midi(input_ids, hop_frame, chunk_frame):
    hop_len = hop_frame // SEG_RES
    chunk_len = chunk_frame // SEG_RES
    seg_pos = torch.arange(len(input_ids[0])).to(input_ids.device)
    seg_pos = seg_pos[input_ids[0, :, 0] == SEG_RES]
    return input_ids[:, :seg_pos[hop_len]], input_ids[:, seg_pos[hop_len]: seg_pos[chunk_len] + 1]



def remove_head(x):
    idx = x[0, :, 0] < RES_EVENT
    return x[0, idx].unsqueeze(0)


class InferenceHelper:
    def __init__(self, model_folder, device):
        from shoelace.actual_shoelace.shoelace import Shoelace as Model
        from shoelace.actual_shoelace.audio_2_midi.config import MODEL_FACTORY
        self.model = Model(mask_type=None, device=torch.device("cuda"), model_configs=MODEL_FACTORY, 
                model_names=["MIDILM", "AudioLM"])
        self.model.load_weights(model_folder)
        self.model.eval().to(device)

    
    @torch.no_grad()
    def inference(self, input_ids_generator, chunk_frame, hop_frame, top_k):
        midi_prompt = None
        n_id = 0
        chunk_len = chunk_frame // SEG_RES
        results = []
        for input_ids, audio_index in input_ids_generator:
            if audio_index is None:
                break
            self.model.inference(model_name="AudioLM", max_len=1, input_ids=input_ids,
                            use_generator=False, top_k=top_k, reset_cache=True,
                            last_chunk=True, device=input_ids.device)


            midi_codes = self.model.inference(model_name="MIDILM", 
                            cond_model_name="AudioLM", max_len=chunk_len,
                            use_generator=True, top_k=top_k, reset_cache=True,
                            last_chunk=False, input_ids=midi_prompt, 
                            cond_indices=audio_index,
                            batch_size=len(input_ids), device=input_ids.device)
            prefix, midi_prompt = cut_midi(midi_codes, hop_frame, chunk_frame)
            results.append(prefix)
            
            n_id += 1
            break
            if n_id > 2:
                break
        
        
        results.append(remove_head(midi_prompt))
        midi_codes = torch.concat(results, 1)
        return midi_codes, input_ids



if __name__=="__main__":
    inference_helper = InferenceHelper()
    input_ids = torch.ones([1, 10, 6]).cuda().long()
    audio_codes = inference_helper.inference(input_ids, max_len=16*50)
    print(audio_codes.shape)
        
            