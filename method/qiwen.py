# api_call_example.py
from openai import OpenAI
import re
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
from gectoolkit.evaluate.cherrant import parallel_to_m2, compare_m2_for_evaluation

'''
运行命令：
conda activate llama_factory
cd /home/xiaoman/project/gec/HandwrittenGEC/gectoolkit/llm/LLaMA-Factory/
API_PORT=8000 CUDA_VISIBLE_DEVICES=6 llamafactory-cli api examples/inference/qwen2_vl.yaml
'''

client = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")

def replace_punctuation(text):
    # 定义英文标点和对应的中文标点
    punctuation_mapping = {
        ',': '，',
        # '.': '。',
        '?': '？',
        '!': '！',
        ";":'；',
        ":":'：'
    }

    # 遍历映射表，替换文本中的标点符号
    for en, zh in punctuation_mapping.items():
        text = text.replace(en, zh)
    return text


def run_qiwen(passage,prompt_template):
    sentences = re.split(r'(?<=[。！？ ?!.])', passage)
    sentences = [replace_punctuation(sentence) for sentence in sentences if sentence.strip()]

    print('----------------------------------')
    for i, sentence in enumerate(sentences):
        print(f'分句{i}:', sentence)
    print('------------------------------')

    corrected_sentences = []

    for sentence in sentences:
        prompt = prompt_template.format(sentence)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": sentence}
        ]
        answer = client.chat.completions.create(messages=messages, model="qiwen7b-visual-text")
        result = answer.choices[0].message.content.strip()
        print('sentence:', sentence)
        print('result  :', result)
        if sentence == result:
            print('没有修改')
        else:
            print('有修改')
        print()
        corrected_sentences.append(result)

    corrected_passage = ''.join(corrected_sentences)

    # 输出最终修正后的文章
    print("纠正后的文章：")
    print(corrected_passage)

    corrected_passage_zh = replace_punctuation(corrected_passage)
    print('corrected_passage_zh', corrected_passage_zh)
    return sentences,corrected_sentences,corrected_passage

def evaluate_m2(sentences, corrected_sentences):
    path = '../gectoolkit/evaluate/cherrant/samples/hyp.para'
    m2_path = '../gectoolkit/evaluate/cherrant/samples/hyp.m2.char'

    with open(path, 'w', encoding='utf-8') as f:
        for source, predict in zip(sentences, corrected_sentences):
            source_predict = source + '\t' + predict
            f.write(source_predict + '\n')

    p2m_hyp_args = parallel_to_m2.Args(file=path, output=m2_path)
    parallel_to_m2.main(p2m_hyp_args)
    print('m2格式')


if __name__ == '__main__':
    prompt_template = (
        """你是一位经验丰富的中文老师，专门纠正学生作文中的语法错误。请根据以下要求进行修正：
    1. 仅修正句子中的语法和用词错误。
    2. 如果句子没有错误，请不要进行任何修改，保持原句不变。
    3. 不要改变句子的结构或原意。

    请按照以下格式输出结果：
    [纠正后的文本内容]
    """
    )


    # passage = '我们的目前社会已经发展的很多十年前真的想不到的发展,那我们十年后或者二十年后呢?肯定发展的更多首先手机,我们目前的手机已经功能很好还有越来越好我们的未来会有更快更多功能的手机在我们的家物也会便利更大比如十几年前我们的家里没有多中多样的家物只有简单一点的家物,但是现在呢?全部有甚至未来会有我们现在想不到的东西我们的教育也有关系首先未来我们得学习的部分更多但也能学习的时候更便利学习因为我们的学习材料更丰富还有肯定用笔记本使用小一点我们都用手机和电脑我们的社会每年发展的很快所以我们按照科技发展的速度要努力活我们的人生我们努力的话没有害怕的东西'
    passage = '未来的科技生活会变得非常惊人。在家里智能家电将变得很普遍。例如,智能电视可以通过语音来控制,还会自动推荐喜欢的节目。此外,自动驾驶汽车也会变得越来越常见,这将会使交通更加安全和高效。医疗技术也会有很大的进步,人们可以使用智能医疗设备在家中监测自己的健康状况,并且与医生进行实时的远程沟通。另外,人工智能将会广泛应用于各个领域。例如,在教育领域,人们可以使用智能教育软件学习各种知识,而无需去传统的教室。在工作中,人工智能也会承担更多的工作任务,从而提高生产效率。虚拟现实技术也会有所发展,人们可以通过虚拟现实设备体验到更加丰富和真实的沉浸式体验,同时,随着技术的进步,人们对于隐私和数据安全的关注也会增加。因此,未来的科技生活也需要更加注重保护个人信息和数据安全的措施。总的来说,未来的科技生活将会给我们带来更多便利和乐趣,但我们也需要在享受科技带来的便利同时,保持警惕并且注重个人隐私和数据安全。'
    # passage = "随着科技的不断发展人们的生活也发生了翻天覆地的变化未来的科技生活将会变得更加便捷智能和舒适首先未来的居住环境将会更加智能化人们居住的房子将装备有最先进的智能家居系统通过语者或手势控制,就能实现家电设备的智能运行和管理,同时房子内部的湿度温度和照明等也将会自动调整,使居民能够享受到最舒适的生活环境其次未来的交通方式将会变得更加智能和高效人们将会驾驶无人驾驶汽车通过智能路线规划和自动驾驶技术,实现出行的自动化和高效化,大大减少交通事故和交通堵塞同时,电动飞行汽车的出现也将使人们的出行更加便捷和快速,缓解城市交通压力,提高出行效率总体而言,未来的科技生活将会给人们带来更利和舒适然而,与此同时我们也需要警惕科技发展可能带来的负面影响譬如对人类关系的冲击个人隐私安全等问题只能在科技与人类充分协同的前提下,未来的科技生活才能真正成为人们美好生活的一部分"
    # passage = '未来的科技生活会变得很有趣!想象一下,你早上醒来,房间的灯光会根据你的心情自动调整,让你感觉更加舒适你走到厨房智能冰箱会提醒你食物的质期,并建议你今天应该吃什么你坐在智能汽车里,它能自己开,你可以放心地做其他事情,比如看书或者和朋友视频聊天回家后,你戴上智能手表,它会监测你的健康状况,并提醒你何时该运动或休息晚上,你和家人一起观看全息影院,仿佛身临其境般地体验电影的奇妙在未来,科技将让我们的生活变得更加便捷有趣让我们享受到前所未有的舒适和乐趣!'

    # passage = '我在上海读书了半年多了。虽然在这生活了一段不短的时间了,但是紧张的学习让我没有时间做我想做的事。之所以我想利用下个月的假日来好好放松一下自己的身心和完成我以前还没完成的计划。这周就是期中考试的,加上考试完后刚好是放五一的,所以我会好好放松放松。首先,我已经和朋友约好下周会一起去打羽毛球。打完羽毛球后我们会一起去吃夜销。其次,我会去参观一下世纪公园。因为我在小红书看到大家的照,拍的特别好看,公园里面还有一个很大的潮,所以我想去看看。另外里面的花也开得特别好,所以我希望,我去的时候它们还没调谢。最后,我打算到时看情况,如果我不累,我会去上海其他公园。然后到最后放假的一天,我会在宿舍好好体息,预习一下,明天要上课的内容,让自己避免得了假期综合征。这样还可以提高我学习效率。这是我今年五一的计划,希望我能开开心心地度过这个假期。'
    # corrected_passage1 = '我在上海读书已有半年多，虽然在这生活了一段时间，但紧张的学习让我没有时间做我想做的事。因此，我计划利用下个月的假期好好放松身心，并完成之前未完成的计划。这周是期中考试，考试结束后正好是五一假期，所以我会好好放松。首先，我已经和朋友约好下周一起去打羽毛球，打完后我们会一起吃夜宵。其次，我打算去参观世纪公园，因为我在小红书上看到大家的美照，公园里有一个很大的湖泊，我想去看看。另外，公园里的花也开得特别好，我希望我去的时候它们还没凋谢。最后，我打算根据情况，如果我不累，我会去上海其他公园。然后，在放假的最后一天，我会在宿舍好好休息，预习明天要上课的内容，避免假期综合征。这样还可以提高我的学习效率。这是我今年五一的计划，希望我能开心地度过这个假期。'


    sentences,correct_sentences,corrected_passage = run_qiwen(passage, prompt_template)
    evaluate_m2(sentences=sentences, corrected_sentences=correct_sentences)