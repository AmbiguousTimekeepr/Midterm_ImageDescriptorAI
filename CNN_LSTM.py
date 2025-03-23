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
        base_dir = r"Datasets\CNN"

        # Construct full image path
        img_path = os.path.join(base_dir, img_path)
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

class VQAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512, pretrained=False):
        super(VQAModel, self).__init__()
        
        self.vocab_size = vocab_size
        # Special tokens - ensure these match your vocabulary
        self.sos_idx = 2  # <SOS> token
        self.eos_idx = 3  # <EOS> token
        self.pad_idx = 0  # <PAD> token
        
        # Image encoder - Using ResNet
        self.cnn = models.resnet18(pretrained=pretrained)
        
        # Question encoder components
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.question_encoder = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            batch_first=True,
            bidirectional=True
        )
        self.question_projection = nn.Linear(hidden_dim*2, hidden_dim)
        
        # Attention mechanism
        self.attention = Attention(512, hidden_dim)
        
        # Encoder fusion layer
        self.encoder_fusion = nn.Sequential(
            nn.Linear(512 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Decoder for answer generation
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
    
    def encode(self, image, question, question_lengths):
        # Process image through CNN layers
        x = self.cnn.conv1(image)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        img_features = self.cnn.layer4(x)  # [batch, 512, 7, 7]
        
        # Reshape for attention
        batch_size = img_features.size(0)
        img_features = img_features.view(batch_size, 512, -1).permute(0, 2, 1)  # [batch, 49, 512]
        
        # Process question
        embedded = self.embedding(question)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, question_lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, cell) = self.question_encoder(packed)
        
        # Combine bidirectional outputs
        question_features = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        question_features = self.question_projection(question_features)
        
        # Apply attention
        context, _ = self.attention(img_features, question_features)
        
        # Create thought vector by fusing features
        thought_vector = self.encoder_fusion(torch.cat((context, question_features), dim=1))
        
        return thought_vector, cell[-1]
    
    def forward(self, image, question, question_lengths, answer=None, teacher_forcing_ratio=0.5):
        batch_size = image.size(0)
        
        # Encode image and question
        thought_vector, memory_cell = self.encode(image, question, question_lengths)
        
        # Prepare for decoding
        target_length = answer.size(1) if answer is not None else 20
        outputs = torch.zeros(batch_size, target_length, self.vocab_size, device=image.device)
        
        # Initialize decoder input with <SOS> token
        decoder_input = torch.full((batch_size, 1), self.sos_idx, 
                                  dtype=torch.long, device=image.device)
        
        # Set initial hidden state with encoded thought vector
        decoder_hidden = (thought_vector.unsqueeze(0), 
                          memory_cell.unsqueeze(0))
        
        # Generate answer sequence
        for t in range(target_length):
            # Get embedding of current input token
            decoder_emb = self.embedding(decoder_input)
            
            # Run through decoder LSTM
            decoder_output, decoder_hidden = self.decoder(decoder_emb, decoder_hidden)
            
            # Project to vocabulary
            prediction = self.output_projection(decoder_output.squeeze(1))
            outputs[:, t] = prediction
            
            # Determine next input: teacher forcing or own prediction
            use_teacher_forcing = random.random() < teacher_forcing_ratio and answer is not None
            
            if use_teacher_forcing:
                decoder_input = answer[:, t].unsqueeze(1)
            else:
                _, top_indices = prediction.topk(1)
                decoder_input = top_indices
                
                # Check if all sequences have generated <EOS>
                if all(decoder_input.eq(self.eos_idx).view(-1)):
                    break
                    
        return outputs
    
    def generate_answer(self, image, question, question_lengths, max_length=20):
        batch_size = image.size(0)
        
        # Encode image and question
        thought_vector, memory_cell = self.encode(image, question, question_lengths)
        
        # Initialize decoder input with <SOS> token
        decoder_input = torch.full((batch_size, 1), self.sos_idx, 
                                  dtype=torch.long, device=image.device)
        
        # Set initial hidden state with encoded thought vector
        decoder_hidden = (thought_vector.unsqueeze(0), 
                          memory_cell.unsqueeze(0))
        
        # Store generated tokens
        generated_tokens = []
        
        # Generate sequence
        for _ in range(max_length):
            decoder_emb = self.embedding(decoder_input)
            decoder_output, decoder_hidden = self.decoder(decoder_emb, decoder_hidden)
            prediction = self.output_projection(decoder_output.squeeze(1))
            
            # Get most likely next token
            _, top_indices = prediction.topk(1)
            token = top_indices.item()
            generated_tokens.append(token)
            
            # Break if <EOS> generated
            if token == self.eos_idx:
                break
                
            # Next input is current prediction
            decoder_input = top_indices
            
        return generated_tokens

def collate_fn(batch):
    # Sort by question length (important for packed sequence processing)
    batch = sorted(batch, key=lambda x: len(x['question']), reverse=True)
    
    # Stack all images into a single tensor
    images = torch.stack([item['image'] for item in batch])
    
    # Get question sequences and their actual lengths (needed for packing)
    questions = [item['question'] for item in batch]
    question_lengths = torch.tensor([len(q) for q in questions])
    
    # Get answer sequences
    answers = [item['answer'] for item in batch]
    
    # Pad sequences to same length within batch
    padded_questions = nn.utils.rnn.pad_sequence(questions, batch_first=True)
    padded_answers = nn.utils.rnn.pad_sequence(answers, batch_first=True)
    
    # Keep original data for reference
    breeds = [item['breed'] for item in batch]
    
    return {
        'image': images,
        'question': padded_questions,
        'question_lengths': question_lengths,
        'answer': padded_answers,
        'breed': breeds
    }

def evaluate(model, dataloader, criterion, device, vocab=None, print_samples=5):
    model = model.to(device)
    model.eval()
    total_loss = 0
    
    # For BLEU score calculation
    all_references = []
    all_hypotheses = []
    
    # Create index-to-word mapping for readable output
    if vocab:
        idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # Track number of samples to display
    samples_shown = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            questions = batch['question'].to(device)
            question_lengths = batch['question_lengths']
            answers = batch['answer'].to(device)
            
            # Forward pass (teacher forcing = 0 for evaluation)
            outputs = model(images, questions, question_lengths, answers, teacher_forcing_ratio=0)
            
            # Handle loss calculation
            outputs = outputs[:, :-1, :]  # Remove last prediction
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            answers_flat = answers[:, 1:].contiguous().view(-1)  # Remove <SOS>
            
            loss = criterion(outputs_flat, answers_flat)
            total_loss += loss.item()
            
            # Get readable predictions for some samples
            if vocab and samples_shown < print_samples:
                # Get argmax for each position in sequence
                output_sequences = outputs.argmax(dim=2)  # [batch, seq_len]
                
                # Print some samples
                for i in range(min(3, output_sequences.size(0))):
                    # Convert question tokens to words
                    q_tokens = questions[i].cpu().numpy()
                    q_words = [idx_to_word.get(idx.item(), '<UNK>') for idx in questions[i]]
                    q_text = ' '.join([w for w in q_words if w not in ['<PAD>', '<SOS>', '<EOS>']])
                    
                    # Convert target answer tokens to words
                    a_tokens = answers[i].cpu().numpy()
                    a_words = [idx_to_word.get(idx.item(), '<UNK>') for idx in answers[i]]
                    a_text = ' '.join([w for w in a_words if w not in ['<PAD>', '<SOS>', '<EOS>']])
                    
                    # Convert predicted answer tokens to words
                    p_tokens = output_sequences[i].cpu().numpy()
                    p_words = []
                    for idx in p_tokens:
                        word = idx_to_word.get(idx, '<UNK>')
                        if word == '<EOS>':
                            break
                        if word not in ['<PAD>', '<SOS>']:
                            p_words.append(word)
                    p_text = ' '.join(p_words)
                    
                    print(f"Question: {q_text}")
                    print(f"Target: {a_text}")
                    print(f"Predicted: {p_text}")
                    print("-" * 50)
                
                samples_shown += 1
            
            # Add to lists for BLEU calculation
            output_sequences = outputs.argmax(dim=2).cpu().numpy()
            
            for i in range(answers.size(0)):
                # Get target tokens (excluding special tokens)
                target_seq = []
                for idx in answers[i].cpu().numpy():
                    if idx != vocab['<PAD>'] and idx != vocab['<SOS>'] and idx != vocab['<EOS>']:
                        target_seq.append(idx_to_word.get(idx, '<UNK>'))
                
                # Get predicted tokens (excluding special tokens, stopping at EOS)
                pred_seq = []
                for idx in output_sequences[i]:
                    word = idx_to_word.get(idx, '<UNK>')
                    if word == '<EOS>':
                        break
                    if word != '<PAD>' and word != '<SOS>':
                        pred_seq.append(word)
                
                all_references.append([target_seq])
                all_hypotheses.append(pred_seq)
                
        bleu_score = 0
        if all_hypotheses:
            bleu_score = bleu.corpus_bleu(all_references, all_hypotheses)
            
    print(f"BLEU-4 Score: {bleu_score:.4f}")
    return total_loss / len(dataloader)

def train_epoch(model, dataloader, optimizer,scheduler, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    
    for batch in dataloader:
        # Move data to device
        images = batch['image'].to(device)
        questions = batch['question'].to(device)
        question_lengths = batch['question_lengths']
        answers = batch['answer'].to(device)
        # Forward pass
        outputs = model(images, questions, question_lengths, answers, teacher_forcing_ratio)
        
        # Calculate loss (exclude first token which is <SOS>)
        # Reshape outputs for loss calculation
        outputs = outputs[:, :-1, :]  # Remove last prediction
        outputs_flat = outputs.contiguous().view(-1, outputs.size(-1))
        answers_flat = answers[:, 1:].contiguous().view(-1) 
        
        loss = criterion(outputs_flat, answers_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_model(train_pairs, val_pairs, vocab, num_epochs=50):
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DogBreedVQADataset(train_pairs, vocab, transform)
    val_dataset = DogBreedVQADataset(val_pairs, vocab, transform)
    
    # Create data loaders with collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        collate_fn=collate_fn
    )
    
    pretrained_model = VQAModel(vocab_size=len(vocab), pretrained=True)
    house_trained_model = VQAModel(vocab_size=len(vocab), pretrained=False)
    
    pretrained_model = train_single_model(pretrained_model, train_loader, val_loader, vocab, num_epochs, trained_name = "pretrained_vqa_model.pth")
    house_trained_model = train_single_model(house_trained_model, train_loader, val_loader, vocab, num_epochs, trained_name = "house-trained_vqa_model.pth")
    
    return pretrained_model, house_trained_model


def train_single_model(model, train_loader, val_loader, vocab, num_epochs, trained_name = "trained_model.pth"):

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001, 
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, criterion, device, vocab)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # concat a new name for each model ()
    torch.save(model.state_dict(), trained_name)
    return model

    
def compare_models(pretrained_model, house_trained_model, test_loader, device):
    pretrained_model = pretrained_model.to(device)
    house_trained_model = house_trained_model.to(device)

    pretrained_model.eval()
    house_trained_model.eval()
    
    pretrained_correct = 0
    house_trained_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            questions = batch['question'].to(device)
            question_lengths = batch['question_lengths']
            answers = batch['answer'].to(device)
            
            pretrained_outputs = pretrained_model(images, questions, question_lengths, answers, teacher_forcing_ratio=0)
            house_trained_outputs = house_trained_model(images, questions, question_lengths, answers, teacher_forcing_ratio=0)
            
            _, pretrained_preds = pretrained_outputs.max(2)
            _, house_trained_preds = house_trained_outputs.max(2)
            
            pretrained_correct += (pretrained_preds == answers).all(dim=1).sum().item()
            house_trained_correct += (house_trained_preds == answers).all(dim=1).sum().item()
            total += answers.size(0)
    
    pretrained_accuracy = pretrained_correct / total
    house_trained_accuracy = house_trained_correct / total
    
    print(f"Pretrained Model Accuracy: {pretrained_accuracy:.4f}")
    print(f"House-trained Model Accuracy: {house_trained_accuracy:.4f}")
    
    if pretrained_accuracy > house_trained_accuracy:
        print("The pretrained model performs better.")
    elif house_trained_accuracy > pretrained_accuracy:
        print("The house-trained model performs better.")
    else:
        print("Both models perform equally.")


def load_qa_data(csv_file):
    qa_pairs = []
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            breed = row['breed']
            image_path = row['image_path']
            question = row['question']
            answer = row['answer']
            qa_pairs.append((breed, image_path, question, answer))
    
    return qa_pairs

def load_qa_data_subset(csv_file, fraction=0.25):
    qa_pairs = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        subset_size = int(len(all_rows) * fraction)
        subset_rows = random.sample(all_rows, subset_size)
        
        for row in subset_rows:
            breed = row['breed']
            image_path = row['image_path']
            question = row['question']
            answer = row['answer']
            qa_pairs.append((breed, image_path, question, answer))
            
    return qa_pairs


def prerequisite():
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<SOS>': 2,
        '<EOS>': 3
    }
    qa_df = pd.read_csv(r'Datasets\external\vqa_dataset_split.csv')

    # Add other words from dataset
    word_idx = 4
    for question, answer in qa_df[['Question', 'Answer']].values:
        for word in question.lower().split():
            if word not in vocab:
                vocab[word] = word_idx
                word_idx += 1
        
        for word in answer.lower().split():
            if word not in vocab:
                vocab[word] = word_idx
                word_idx += 1
                
    return vocab