from graphviz import Digraph

def generate_vae_diagram():
    # Create a Digraph instance
    diagram = Digraph('VAE Architecture', format='png')
    diagram.attr(rankdir='LR', size='28')

    # Add nodes for input and output
    diagram.node('Input', 'Input Image (3x128x128)', shape='box', style='filled', fillcolor='#b3cde3')
    diagram.node('Reconstructed', 'Reconstructed Image (3x128x128)', shape='box', style='filled', fillcolor='#b3cde3')

    # Add nodes for encoder layers
    diagram.node('EncConv1', 'Conv2d(3, 32, 4, 2, 1)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('EncConv2', 'Conv2d(32, 64, 4, 2, 1)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('EncConv3', 'Conv2d(64, 128, 4, 2, 1)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('Flatten', 'Flatten', shape='ellipse', style='filled', fillcolor='#fbb4ae')
    
    # Latent space
    diagram.node('Mu', '\u03bc (Mean)', shape='ellipse', style='filled', fillcolor='#decbe4')
    diagram.node('LogVar', '\u03c3\u00b2 (Log Variance)', shape='ellipse', style='filled', fillcolor='#decbe4')
    diagram.node('Z', 'z (Latent Space)', shape='ellipse', style='filled', fillcolor='#decbe4')

    # Add nodes for decoder layers
    diagram.node('DecFC', 'Linear(Latent_dim, 128x16x16)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('DecDeconv1', 'ConvTranspose2d(128, 64, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('DecDeconv2', 'ConvTranspose2d(64, 32, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('DecDeconv3', 'ConvTranspose2d(32, 3, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')

    # Add edges to represent data flow
    diagram.edges([('Input', 'EncConv1'), ('EncConv1', 'EncConv2'), ('EncConv2', 'EncConv3'), ('EncConv3', 'Flatten')])
    diagram.edges([('Flatten', 'Mu'), ('Flatten', 'LogVar'), ('Mu', 'Z'), ('LogVar', 'Z')])
    diagram.edge('Z', 'DecFC')
    diagram.edges([('DecFC', 'DecDeconv1'), ('DecDeconv1', 'DecDeconv2'), ('DecDeconv2', 'DecDeconv3'), ('DecDeconv3', 'Reconstructed')])

    # Render diagram
    diagram.render('vae_architecture_diagram', view=True)

def generate_gan_diagram():
    # Create a Digraph instance
    diagram = Digraph('GAN Architecture', format='png')
    diagram.attr(rankdir='TB', size='16')

    # Generator components
    diagram.node('Latent', 'Latent Vector (z)', shape='ellipse', style='filled', fillcolor='#decbe4')
    diagram.node('GConv1', 'ConvTranspose2d(100, 512, 4, 1, 0)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('GConv2', 'ConvTranspose2d(512, 256, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('GConv3', 'ConvTranspose2d(256, 128, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('GConv4', 'ConvTranspose2d(128, 64, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('GConv5', 'ConvTranspose2d(64, 3, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('Generated', 'Generated Image (3x64x64)', shape='ellipse', style='filled', fillcolor='#b3cde3')

    # Discriminator components
    diagram.node('DInput', 'Input Image (3x64x64)', shape='ellipse', style='filled', fillcolor='#b3cde3')
    diagram.node('DConv1', 'Conv2d(3, 128, 4, 2, 1)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('DConv2', 'Conv2d(128, 256, 4, 2, 1)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('DConv3', 'Conv2d(256, 512, 4, 2, 1)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('DConv4', 'Conv2d(512, 1, 4, 1, 0)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('Sigmoid', 'Sigmoid Activation', shape='ellipse', style='filled', fillcolor='#fbb4ae')
    diagram.node('Output', 'Real/Fake Output', shape='ellipse', style='filled', fillcolor='#decbe4')

    # Generator connections
    diagram.edges([('Latent', 'GConv1'), ('GConv1', 'GConv2'), ('GConv2', 'GConv3'), 
                   ('GConv3', 'GConv4'), ('GConv4', 'GConv5'), ('GConv5', 'Generated')])

    # Discriminator connections
    diagram.edges([('DInput', 'DConv1'), ('DConv1', 'DConv2'), ('DConv2', 'DConv3'), 
                   ('DConv3', 'DConv4'), ('DConv4', 'Sigmoid'), ('Sigmoid', 'Output')])

    # Render the diagram
    diagram.render('gan_architecture_diagram', view=True)

def generate_conditional_gan_diagram():
    # Create a Digraph instance
    diagram = Digraph('Conditional GAN Architecture', format='png')
    diagram.attr(rankdir='LR', size='28')

    # Generator components
    diagram.node('Noise', 'Noise Vector (z)', shape='ellipse', style='filled', fillcolor='#b3cde3')
    diagram.node('LabelG', 'Label Embedding', shape='ellipse', style='filled', fillcolor='#b3cde3')
    diagram.node('ConcatG', 'Concatenate (z, Label)', shape='ellipse', style='filled', fillcolor='#decbe4')

    diagram.node('G1', 'ConvTranspose2d(z+Label, 512, 4, 1, 0)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('G2', 'ConvTranspose2d(512, 256, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('G3', 'ConvTranspose2d(256, 128, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('G4', 'ConvTranspose2d(128, 64, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('G5', 'ConvTranspose2d(64, 3, 4, 2, 1)', shape='box', style='filled', fillcolor='#ccebc5')
    diagram.node('OutputG', 'Generated Image (3x64x64)', shape='ellipse', style='filled', fillcolor='#b3cde3')

    # Discriminator components
    diagram.node('ImageD', 'Input Image (3x64x64)', shape='ellipse', style='filled', fillcolor='#b3cde3')
    diagram.node('LabelD', 'Label Embedding', shape='ellipse', style='filled', fillcolor='#b3cde3')
    diagram.node('ConcatD', 'Concatenate (Image, Label)', shape='ellipse', style='filled', fillcolor='#decbe4')

    diagram.node('D1', 'Conv2d(Image+Label, 128, 4, 2, 1)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('D2', 'Conv2d(128, 256, 4, 2, 1)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('D3', 'Conv2d(256, 512, 4, 2, 1)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('D4', 'Conv2d(512, 1, 4, 1, 0)', shape='box', style='filled', fillcolor='#fbb4ae')
    diagram.node('OutputD', 'Real/Fake Prediction', shape='ellipse', style='filled', fillcolor='#b3cde3')

    # Generator edges
    diagram.edges([
        ('Noise', 'ConcatG'), 
        ('LabelG', 'ConcatG'), 
        ('ConcatG', 'G1'), ('G1', 'G2'), ('G2', 'G3'), ('G3', 'G4'), ('G4', 'G5'), ('G5', 'OutputG')
    ])

    # Discriminator edges
    diagram.edges([
        ('ImageD', 'ConcatD'),
        ('LabelD', 'ConcatD'),
        ('ConcatD', 'D1'), ('D1', 'D2'), ('D2', 'D3'), ('D3', 'D4'), ('D4', 'OutputD')
    ])

    # Render the diagram
    diagram.render('conditional_gan_architecture_diagram', view=True)

# Generate the VAE architecture diagram
generate_vae_diagram()

# Generate the GAN architecture diagram
generate_gan_diagram()

# Generate the Conditional GAN architecture diagram
generate_conditional_gan_diagram()
