# contributed by: https://github.com/ayushp88 (Ayush Pandey)

import re

import torch

from src.methods.soft_prompting import SoftPromptTuning
from src.utils import (
    get_tokenizer,
    get_tuned_model_path,
    FineTuningType,
    get_base_model,
)


class SoftPromptTester:
    def __init__(
        self,
        base_model_name=get_base_model(),
        soft_prompt_path=get_tuned_model_path(FineTuningType.SOFT_PROMPTS),
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = get_tokenizer(base_model_name)

        self.model = SoftPromptTuning(
            device=self.device,
            tokenizer=self.tokenizer,
            model_name=base_model_name,
        )
        self.load_soft_prompts(soft_prompt_path)
        self.model.to(self.device)
        self.model.eval()

    def load_soft_prompts(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if "soft_prompts.weight" in checkpoint:
            self.model.soft_prompts.weight.data.copy_(checkpoint["soft_prompts.weight"])
            print("Soft prompts loaded successfully.")
        else:
            raise KeyError("Key 'soft_prompts.weight' not found in checkpoint.")

    def clean_generated_text(self, text):
        """Remove unwanted Markdown-like characters from the generated text."""
        # Remove Markdown headers and asterisks
        text = re.sub(r"^[#*]+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*+", "", text)
        return text.strip()

    def generate_summary(self, text, max_summary_tokens=150):
        """Generate a summary using the loaded soft prompt model."""
        # Tokenize input text
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt", max_length=1024, truncation=True
        ).to(self.device)

        # Prepend soft prompts to input embeddings
        soft_prompt_ids = torch.arange(self.model.num_soft_prompts, device=self.device)
        soft_prompt_embeds = self.model.soft_prompts(soft_prompt_ids).unsqueeze(0)
        input_embeds = self.model.gpt_neo.transformer.wte(input_ids)
        inputs_embeds = torch.cat([soft_prompt_embeds, input_embeds], dim=1)

        # Adjust attention mask to include soft prompts
        attention_mask = torch.ones(inputs_embeds.size()[:-1], device=self.device)

        # Generate summary
        outputs = self.model.gpt_neo.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_summary_tokens,
            num_beams=4,
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode and clean the summary
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.clean_generated_text(summary)


if __name__ == "__main__":
    tester = SoftPromptTester(
        base_model_name="EleutherAI/gpt-neo-125M",
        soft_prompt_path="res/soft_prompts.pth",
    )

    # test paragraph
    test_paragraph = """
   If you're famous and performing the American national anthem, be prepared to become a national hero or a national disgrace. Facts are facts. Just ask Vince, Whitney, Roseanne, Jimi and Michael. Mötley Crüe's Vince Neil reminded us again this week of the dangers of tackling "The Star-Spangled Banner." Sure, he can shred it on "Girls, Girls, Girls" and "Dr. Feelgood," but this is a different story -- a completely different story. To say Neil butchered the song before the Las Vegas Outlaws Arena Football League game would be unkind to those in the profession. There's less carnage when butchers are done with their work. The late Whitney Houston set the modern standard for the national anthem at Super Bowl XXV. In the early stages of the Gulf War in 1991, a patriotic America saluted her performance. Just six months earlier, comedian Roseanne Barr may have established the low-water mark. The crowd at the San Diego Padres game booed her rendition and President George H. W. Bush called it "disgraceful." There's nothing quite like getting the presidential thumbs down. One of the most controversial and beloved versions of "The Star-Spangled Banner" comes from 1969. Guitar slinger Jimi Hendrix inflamed mainstream America with his psychedelic take on the national anthem to the delight of the Woodstock generation. And then there's Michael Bolton's version. Overly wrought songs are his specialty and he doesn't disappoint in that department when he sings at the American League Championship Series in 2003. Bolton belts it out, but there's one little problem -- the words. Can anyone say crib notes?
    """

    print("Generating summary...")
    summary = tester.generate_summary(test_paragraph, max_summary_tokens=150)
    print("\nOriginal Paragraph:")
    print(test_paragraph)
    print("\nGenerated Summary:")
    print(summary)
