from setuptools import setup, find_packages
import os

def read_file(filename):
    filepath = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), filename)
    if os.path.exists(filepath):
        return open(filepath).read()
    else:
        return ''

setup(name='vocoder_eva',
      version='0.1',
      description='evaluate vocoder',
      long_description=read_file('readme.md'),
      long_description_content_type="text/markdown",
      url='https://github.com/exeex/vocoder_eva',
      author='cswu',
      author_email='xray0h@gmail.com',
      license='Apache License 2.0',
      packages=find_packages(exclude=['assets', 'target_projects']),
      zip_safe=False,
      # test_suite='tests',
      install_requires= [
        'librosa',
        # 'pyworld'
      ],
      # entry_points = {
      #   'console_scripts': ['lgl=lgl.lgl:main'],
      # },
      classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Utilities",
      ],
)
