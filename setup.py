from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='SSNMRPY',
  version='0.0.1',
  description='Denoising SSNMR Spectra',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Adam Altenhof',
  author_email='adamaltenhof@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='NMR', 
  packages=find_packages(),
  install_requires=['numpy','scipy'] 
)