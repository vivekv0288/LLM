!git config --global user.name "vish02chouhan"
!git config --global user.email "vish02chouhan@gmail.com"
!git clone https://github.com/vivekv0288/LLM.git
%cd LLM
!git checkout development

!git add readme.txt
!git commit -m 'generate model'

!git pull --rebase

!git push https://vish02chouhan:access_token@github.com/vivekv0288/LLM.git