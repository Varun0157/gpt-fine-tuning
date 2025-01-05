import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn


class SoftPromptTuning:
    def __init__(self, base_model_name="gpt2", soft_prompt_path="/data4/ayush/prompt_tune/res/soft_prompts.pth", num_soft_prompts=12):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(base_model_name).to(self.device)

        # Load soft prompts
        self.num_soft_prompts = num_soft_prompts
        self.embedding_dim = self.model.config.hidden_size
        self.soft_prompts = nn.Embedding(self.num_soft_prompts, self.embedding_dim).to(self.device)
        self.load_soft_prompts(soft_prompt_path)

    def load_soft_prompts(self, path):
        """Load the trained soft prompt weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.soft_prompts.weight.data.copy_(checkpoint['soft_prompts.weight'])
        print("Soft prompts loaded successfully.")

    def generate_summary(self, text, max_summary_tokens=150):
        """Generate a summary using the soft prompt model."""
        # Tokenize input text
        input_ids = self.tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)

        # Get embeddings for input tokens
        input_embeds = self.model.transformer.wte(input_ids)

        # Add soft prompts to input embeddings
        soft_prompt_ids = torch.arange(self.num_soft_prompts, device=self.device)
        soft_prompt_embeds = self.soft_prompts(soft_prompt_ids).unsqueeze(0)
        inputs_embeds = torch.cat([soft_prompt_embeds, input_embeds], dim=1)

        # Adjust attention mask to include soft prompts
        attention_mask = torch.ones(inputs_embeds.size()[:-1], device=self.device)

        # Generate summary
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_summary_tokens,
            num_beams=4,
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode and return the summary
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()


# Example Usage
if __name__ == "__main__":
    # Initialize the soft prompt tuning model
    soft_prompt_model = SoftPromptTuning(
        base_model_name="gpt2",
        soft_prompt_path="/data4/ayush/prompt_tune/res/soft_prompts.pth",
        num_soft_prompts=12
    )

    # Test paragraph
    test_paragraph = """
    A student has admitted to hanging a noose made of rope from a tree near a student union, university officials said Thursday. The prestigious private school didn't identify the student, citing federal privacy laws. In a news release, it said the student was no longer on campus and will face student conduct review. The student was identified during an investigation by campus police and the office of student affairs and admitted to placing the noose on the tree early Wednesday, the university said. Officials are still trying to determine if other people were involved. Criminal investigations into the incident are ongoing as well. Students and faculty members marched Wednesday afternoon chanting "We are not afraid. We stand together," after pictures of the noose were passed around on social media. At a forum held on the steps of Duke Chapel, close to where the noose was discovered at 2 a.m., hundreds of people gathered. "You came here for the reason that you want to say with me, 'This is no Duke we will accept. This is no Duke we want. This is not the Duke we're here to experience. And this is not the Duke we're here to create,' " Duke President Richard Brodhead told the crowd. The incident is one of several recent racist events to affect college students. Last month a fraternity at the University of Oklahoma had its charter removed after a video surfaced showing members using the N-word and referring to lynching in a chant. Two students were expelled. In February, a noose was hung around the neck of a statue of a famous civil rights figure at the University of Mississippi. A statement issued by Duke said there was a previous report of hate speech directed at students on campus. In the news release, the vice president for student affairs called the noose incident a "cowardly act." "To whomever committed this hateful and stupid act, I just want to say that if your intent was to create fear, it will have the opposite effect," Larry Moneta said Wednesday. Duke University is a private college with about 15,000 students in Durham, North Carolina. CNN's Dave Alsup contributed to this report.
    """

    # Generate summary
    print("Generating summary...")
    summary = soft_prompt_model.generate_summary(test_paragraph, max_summary_tokens=150)
    print("\nOriginal Paragraph:")
    print(test_paragraph)
    print("\nGenerated Summary:")
    print(summary)
