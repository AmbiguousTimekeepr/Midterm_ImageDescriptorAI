from dependencies import *

class DogBreedVQADataset(Dataset):
    def __init__(self, qa_pairs,vocab, transform=None):
        self.qa_pairs = qa_pairs  # (question, answer) tuples
        self.transform = transform
        self.vocab = vocab
        
    def __len__(self):
        return len(self.qa_pairs)
        
    def __getitem__(self, idx):
        breed, img_path, question, answer = self.qa_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'question': self.tokenize(question),
            'answer': self.tokenize(answer),
            'breed': breed
        }
        
    def tokenize(self,string):
        # Add special tokens
        tokens = ['<SOS>'] + string.lower().split() + ['<EOS>']
        # Convert to indices and handle unknown words
        return torch.tensor([self.vocab.get(word, self.vocab['<UNK>']) for word in tokens])

class Attention(nn.Module):
    def __init__(self, image_dim, question_dim, attention_dim=512):
        super(Attention, self).__init__()
        
        # Project both image and question features to the same attention space
        self.image_projection = nn.Linear(image_dim, attention_dim)
        self.question_projection = nn.Linear(question_dim, attention_dim)
        
        # Attention vector for computing weights
        self.attention_vector = nn.Linear(attention_dim, 1)
        
    def forward(self, image_features, question_features):
        # image_features: [batch_size, num_regions, image_dim]
        # question_features: [batch_size, question_dim]
        
        batch_size = image_features.size(0)
        num_regions = image_features.size(1)
        
        # Expand question features to match spatial regions of image
        question_features = question_features.unsqueeze(1).repeat(1, num_regions, 1)
        
        # Project to common attention space
        img_proj = self.image_projection(image_features)  # [batch, regions, attention_dim]
        ques_proj = self.question_projection(question_features)  # [batch, regions, attention_dim]
        
        # Joint attention features
        joint_features = torch.tanh(img_proj + ques_proj)  # [batch, regions, attention_dim]
        
        # Calculate attention scores
        attention_scores = self.attention_vector(joint_features).squeeze(-1)  # [batch, regions]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, regions]
        
        # Apply attention weights to get context vector
        context = torch.sum(attention_weights.unsqueeze(-1) * image_features, dim=1)  # [batch, image_dim]
        
        return context, attention_weights