# Disentangled Sequential Autoencoder
PyTorch implementation of [Disentangled Sequential Autoencoder](https://arxiv.org/abs/1803.02991) (Mandt et al.), a Variational Autoencoder Architecture for learning latent representations of high dimensional sequential data by approximately disentangling the time invariant and the time variable features. 

## Results
We test our network on the [Liberated Pixel Cup](https://github.com/jrconway3/Universal-LPC-spritesheet) dataset consisting of sprites of video game characters of varying hairstyle, clothing, skin color and pose. We constrain ourselves to three particular types of poses, walking, slashing and spellcasting. The network learns disentangled vector representations for the static (elements like skin color and hair color) and dynamic aspects (motion) in the vectors f, and z1, z2, z3, .. z8 (one for each frame), respectively

### Style Transfer
We perform style transfer by learning the f and z  encodings of two characters that differ in both appearance and pose, and swap their z encodings. This causes the characters to interchange their pattern of motion while preserving appearance ,allowing manipulations like "blue dark elf walking" swapped with "lightskinned human spellcasting" gives "blue dark elf spellcasting" and "lightskinned human walking" respectively

<table align='center'>
<tr align='center'>
<td>Sprite 1</td>
<td>Sprite 2</td>
<td>Sprite 1's Body With Sprite 2's Pose</td>
<td>Sprite 2's Body With Sprite 1's Pose</td>
</tr>
<tr>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set1/image1.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set1/image2.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set1/image1_body_image2_motion.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set1/image2_body_image1_motion.png'></td>
</tr>
<tr>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set2/image1.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set2/image2.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set2/image1_body_image2_motion.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set2/image2_body_image1_motion.png'></td>
</tr>
<tr>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set3/image1.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set3/image2.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set3/image1_body_image2_motion.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set3/image2_body_image1_motion.png'></td>
</tr>
<tr>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set4/image1.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set4/image2.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set4/image1_body_image2_motion.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set4/image2_body_image1_motion.png'></td>
</tr>
<tr>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set5/image1.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set5/image2.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set5/image1_body_image2_motion.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set5/image2_body_image1_motion.png'></td>
</tr>
<tr>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set6/image1.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set6/image2.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set6/image1_body_image2_motion.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set6/image2_body_image1_motion.png'></td>
</tr>
<tr>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set7/image1.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set7/image2.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set7/image1_body_image2_motion.png'></td>
<td height="200%"><img height="100% width="150%" src='test/style-transfer/set7/image2_body_image1_motion.png'></td>
</tr>
</table>
