from typing import Dict, Any, List, Optional
import re

from app.utils.logger import logger


class PromptOptimizer:
    """
    提示词优化器
    通过应用各种优化规则增强提示词，提高大模型响应质量
    """
    
    def __init__(self):
        """初始化提示词优化器"""
        # 预定义的优化规则
        self.optimization_rules = [
            self._rule_add_specificity,
            self._rule_adjust_length,
            self._rule_improve_structure,
            self._rule_enhance_instructions,
            self._rule_remove_redundancy
        ]
        logger.info("Prompt optimizer initialized")
    
    def optimize(self, 
                prompt: str, 
                context: Optional[Dict[str, Any]] = None,
                apply_rules: Optional[List[str]] = None,
                target_length: Optional[int] = None) -> Dict[str, Any]:
        """
        对提示词应用优化规则
        
        Args:
            prompt: 要优化的原始提示词
            context: 上下文信息，如内容类型、期望输出格式等
            apply_rules: 要应用的规则列表，如果为None则应用所有规则
            target_length: 目标提示词长度，如果为None则不调整长度
            
        Returns:
            Dict[str, Any]: 包含优化结果的字典
        """
        try:
            # 初始化跟踪变量
            original_prompt = prompt
            optimized_prompt = prompt
            applied_rules = []
            
            # 准备上下文，如果未提供则使用空字典
            context = context or {}
            
            # 确定要应用的规则
            if apply_rules is not None:
                rule_funcs = []
                for rule_name in apply_rules:
                    rule_method = getattr(self, f"_rule_{rule_name}", None)
                    if rule_method is not None:
                        rule_funcs.append(rule_method)
                    else:
                        logger.warning(f"Unknown rule: {rule_name}")
            else:
                rule_funcs = self.optimization_rules
            
            # 如果提供了目标长度，添加目标长度到上下文
            if target_length:
                context["target_length"] = target_length
                
            # 逐一应用规则
            for rule_func in rule_funcs:
                rule_name = rule_func.__name__[6:]  # 去掉"_rule_"前缀
                
                try:
                    result = rule_func(optimized_prompt, context)
                    if result["modified"]:
                        optimized_prompt = result["prompt"]
                        applied_rules.append({
                            "name": rule_name,
                            "impact": result["impact"]
                        })
                        logger.debug(f"Applied rule {rule_name} with impact: {result['impact']}")
                except Exception as e:
                    logger.error(f"Error applying rule {rule_name}: {str(e)}")
            
            # 计算变化统计
            length_change = len(optimized_prompt) - len(original_prompt)
            length_change_percent = (length_change / max(1, len(original_prompt))) * 100
            
            return {
                "success": True,
                "original_prompt": original_prompt,
                "optimized_prompt": optimized_prompt,
                "applied_rules": applied_rules,
                "stats": {
                    "original_length": len(original_prompt),
                    "optimized_length": len(optimized_prompt),
                    "length_change": length_change,
                    "length_change_percent": length_change_percent,
                    "rule_count": len(applied_rules)
                }
            }
            
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original_prompt": prompt,
                "optimized_prompt": prompt  # 返回原始提示词
            }
    
    def rewrite_prompt(self, prompt: str, style: str) -> Dict[str, Any]:
        """
        按照特定风格重写提示词
        
        Args:
            prompt: 原始提示词
            style: 风格类型，如"concise"、"detailed"、"technical"等
            
        Returns:
            Dict[str, Any]: 包含重写结果的字典
        """
        try:
            if style == "concise":
                # 简洁风格：去除冗余词汇，简化句子结构
                new_prompt = self._rewrite_concise(prompt)
                style_description = "Concise style with direct, simplified instructions"
            elif style == "detailed":
                # 详细风格：添加更多说明和细节
                new_prompt = self._rewrite_detailed(prompt)
                style_description = "Detailed style with comprehensive instructions and context"
            elif style == "technical":
                # 技术风格：使用更专业的词汇和结构
                new_prompt = self._rewrite_technical(prompt)
                style_description = "Technical style with specialized terminology"
            elif style == "educational":
                # 教育风格：强调学习目标和结构化输出
                new_prompt = self._rewrite_educational(prompt)
                style_description = "Educational style focusing on learning outcomes"
            else:
                logger.warning(f"Unknown rewrite style: {style}, returning original")
                return {
                    "success": False,
                    "error": f"Unknown style: {style}",
                    "original_prompt": prompt,
                    "rewritten_prompt": prompt
                }
            
            return {
                "success": True,
                "original_prompt": prompt,
                "rewritten_prompt": new_prompt,
                "style": style,
                "style_description": style_description,
                "length_change": len(new_prompt) - len(prompt)
            }
            
        except Exception as e:
            logger.error(f"Error rewriting prompt: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original_prompt": prompt,
                "rewritten_prompt": prompt
            }
    
    def _rewrite_concise(self, prompt: str) -> str:
        """
        以简洁风格重写提示词
        
        Args:
            prompt: 原始提示词
            
        Returns:
            str: 重写后的提示词
        """
        # 去除冗余修饰词
        simplified = re.sub(r'\b(please|kindly|I want you to|I would like you to)\b', '', prompt)
        
        # 使句子更直接
        simplified = re.sub(r'Could you (please )?', '', simplified)
        simplified = re.sub(r'Can you (please )?', '', simplified)
        
        # 将"I want"类句式转换为指令
        simplified = re.sub(r'I want .* to', '', simplified)
        
        # 去除多余空格
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        
        # 分割成段落
        paragraphs = simplified.split('\n\n')
        
        # 对每个段落进行处理
        concise_paragraphs = []
        for para in paragraphs:
            # 如果段落很长，尝试缩短
            if len(para) > 100:
                # 保留前两句和最后一句
                sentences = para.split('. ')
                if len(sentences) > 3:
                    para = '. '.join(sentences[:2]) + '. ... ' + sentences[-1]
            concise_paragraphs.append(para)
        
        # 重新组合
        return '\n\n'.join(concise_paragraphs)
    
    def _rewrite_detailed(self, prompt: str) -> str:
        """
        以详细风格重写提示词
        
        Args:
            prompt: 原始提示词
            
        Returns:
            str: 重写后的提示词
        """
        # 添加更明确的任务描述
        if not prompt.startswith("I need you to"):
            prompt = "I need you to carefully analyze the following content and " + prompt
            
        # 添加期望输出描述（如果没有）
        if "output" not in prompt.lower() and "format" not in prompt.lower():
            prompt += "\n\nPlease provide a comprehensive response that addresses all aspects of the content."
            
        # 添加结构化说明
        if "structure" not in prompt.lower() and "format" not in prompt.lower():
            prompt += "\n\nOrganize your response in a clear, structured manner with appropriate sections and emphasis on key points."
        
        return prompt
    
    def _rewrite_technical(self, prompt: str) -> str:
        """
        以技术风格重写提示词
        
        Args:
            prompt: 原始提示词
            
        Returns:
            str: 重写后的提示词
        """
        # 添加技术风格前缀
        prompt = "Perform a technical analysis of the following content:\n\n" + prompt
        
        # 替换常用词为更专业的术语
        replacements = {
            "look at": "examine",
            "show": "demonstrate",
            "tell": "elaborate",
            "use": "utilize",
            "make": "construct",
            "find": "identify",
            "check": "verify",
            "fix": "resolve",
            "picture": "diagram",
            "talk about": "discuss"
        }
        
        for common, technical in replacements.items():
            prompt = re.sub(r'\b' + common + r'\b', technical, prompt, flags=re.IGNORECASE)
            
        # 添加技术风格的输出指导
        prompt += "\n\nEnsure your analysis addresses the technical components systematically, including appropriate terminology, methodological considerations, and quantitative assessment where applicable."
        
        return prompt
    
    def _rewrite_educational(self, prompt: str) -> str:
        """
        以教育风格重写提示词
        
        Args:
            prompt: 原始提示词
            
        Returns:
            str: 重写后的提示词
        """
        # 添加教育风格前缀
        prompt = "This is an educational exercise. Based on the following content:\n\n" + prompt
        
        # 添加学习目标
        prompt += "\n\nLearning objectives:\n1. Understand the key concepts presented in the material\n2. Apply these concepts in practical contexts\n3. Develop critical thinking about the subject matter"
        
        # 添加教育评估组件
        prompt += "\n\nInclude in your response:\n- Key terms and definitions\n- Main principles or theories\n- Practical applications\n- At least one example problem with solution\n- A self-assessment question to test understanding"
        
        return prompt
    
    def _rule_add_specificity(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        规则：增强提示词的具体性和清晰度
        
        Args:
            prompt: 提示词
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 规则应用结果
        """
        original = prompt
        modified = False
        impact = "none"
        
        # 检查是否已经足够具体
        specificity_score = self._calculate_specificity(prompt)
        
        if specificity_score < 0.5:  # 具体性不足
            # 获取内容类型（如果上下文中提供）
            content_type = context.get("content_type", "unknown")
            
            # 基于内容类型添加特定说明
            if content_type == "formula":
                if "variables" not in prompt.lower():
                    prompt += "\n\nFor any formulas, explain the meaning of each variable and constant."
                    modified = True
                    impact = "medium"
            elif content_type == "chart" or content_type == "diagram":
                if "trends" not in prompt.lower() and "patterns" not in prompt.lower():
                    prompt += "\n\nIdentify and explain key trends, patterns or relationships shown in the visual elements."
                    modified = True
                    impact = "medium"
            elif content_type == "table":
                if "data" not in prompt.lower():
                    prompt += "\n\nAnalyze the data presented in the table, including any notable patterns or outliers."
                    modified = True
                    impact = "medium"
            
            # 通用增强
            if "output format" not in prompt.lower() and "format" not in prompt.lower():
                prompt += "\n\nPlease structure your response with clear headings and organized sections."
                modified = True
                impact = "low"
        
        return {
            "prompt": prompt,
            "modified": modified,
            "impact": impact,
            "original": original
        }
    
    def _rule_adjust_length(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        规则：调整提示词长度，过长则精简，过短则增强
        
        Args:
            prompt: 提示词
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 规则应用结果
        """
        original = prompt
        modified = False
        impact = "none"
        
        # 获取目标长度（如果上下文中提供）
        target_length = context.get("target_length")
        
        # 如果未指定目标长度，根据内容类型估计合适长度
        if target_length is None:
            content_type = context.get("content_type", "unknown")
            if content_type in ["formula", "chart", "diagram"]:
                target_length = 500  # 较短提示适合图表类内容
            else:
                target_length = 1000  # 通用长度
        
        current_length = len(prompt)
        
        # 如果提示词过长，尝试精简
        if current_length > target_length * 1.5:
            # 拆分成段落
            paragraphs = prompt.split('\n\n')
            
            # 精简每个段落
            shortened_paragraphs = []
            for para in paragraphs:
                # 去除冗余短语
                para = re.sub(r'\b(As you can see|As mentioned earlier|It is worth noting that)\b', '', para)
                
                # 如果段落仍然很长，保留关键句子
                if len(para) > 200:
                    sentences = para.split('. ')
                    if len(sentences) > 5:
                        # 保留前2句和最后1句
                        para = '. '.join(sentences[:2]) + '. ... ' + sentences[-1]
                
                shortened_paragraphs.append(para)
            
            # 重新组合
            prompt = '\n\n'.join(shortened_paragraphs)
            
            modified = True
            impact = "high"
            
        # 如果提示词过短，尝试增强
        elif current_length < target_length * 0.5:
            # 添加详细说明
            if "context" not in prompt.lower():
                prompt += "\n\nConsider the complete context provided and make connections between different elements in the content."
            
            # 添加输出期望
            if "detail" not in prompt.lower():
                prompt += "\n\nProvide detailed explanations and concrete examples where appropriate."
                
            modified = True
            impact = "medium"
        
        return {
            "prompt": prompt,
            "modified": modified,
            "impact": impact,
            "original": original
        }
    
    def _rule_improve_structure(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        规则：改进提示词的整体结构和组织
        
        Args:
            prompt: 提示词
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 规则应用结果
        """
        original = prompt
        modified = False
        impact = "none"
        
        # 检查是否有明确的任务描述开头
        has_clear_task = any(prompt.strip().startswith(task) for task in [
            "Analyze", "Explain", "Summarize", "Create", "Describe", "Evaluate", 
            "Generate", "Identify", "List", "Provide", "你是", "你现在是", "请"
        ])
        
        # 检查是否有清晰的分段
        has_paragraphs = prompt.count('\n\n') >= 1
        
        # 检查是否有明确的输出格式指导
        has_format_guidance = "format" in prompt.lower() or "structure" in prompt.lower()
        
        # 1. 添加任务描述开头（如果需要）
        if not has_clear_task:
            content_type = context.get("content_type", "educational content")
            prompt = f"Analyze the following {content_type} and provide insights:\n\n{prompt}"
            modified = True
            impact = "medium"
        
        # 2. 改进段落结构（如果需要）
        if not has_paragraphs:
            # 尝试按逻辑分段
            sections = []
            if "内容" in prompt or "content" in prompt.lower():
                # 分离内容和指令
                parts = re.split(r'(请|please|下列)', prompt, flags=re.IGNORECASE)
                if len(parts) > 1:
                    content_part = parts[0].strip()
                    instruction_part = ''.join(parts[1:]).strip()
                    sections = [content_part, instruction_part]
                    
            if not sections:  # 如果上面的分离不成功
                # 尝试根据句子长度和类型分段
                sentences = re.split(r'([.。!！?？])\s*', prompt)
                current_section = []
                
                for i in range(0, len(sentences), 2):
                    if i < len(sentences):
                        sentence = sentences[i]
                        punctuation = sentences[i+1] if i+1 < len(sentences) else ""
                        current_section.append(sentence + punctuation)
                        
                        # 当累积一定数量的句子或遇到结束句时分段
                        if len(current_section) >= 3 or any(kw in sentence.lower() for kw in ["therefore", "in conclusion", "finally", "总之", "最后"]):
                            sections.append(' '.join(current_section))
                            current_section = []
                
                if current_section:
                    sections.append(' '.join(current_section))
            
            if sections:
                prompt = '\n\n'.join(sections)
                modified = True
                impact = "medium"
        
        # 3. 添加输出格式指导（如果需要）
        if not has_format_guidance:
            content_type = context.get("content_type", "general")
            output_format = context.get("output_format", "structured")
            
            format_guidance = ""
            if output_format == "structured":
                format_guidance = "\n\nOrganize your response with clear headings and sections."
            elif output_format == "json":
                format_guidance = "\n\nProvide your response in JSON format."
            elif output_format == "markdown":
                format_guidance = "\n\nFormat your response using Markdown with appropriate headings and formatting."
            else:
                format_guidance = "\n\nEnsure your response is clear, concise, and well-structured."
                
            prompt += format_guidance
            modified = True
            impact = "low"
        
        return {
            "prompt": prompt,
            "modified": modified,
            "impact": impact,
            "original": original
        }
    
    def _rule_enhance_instructions(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        规则：增强指令的清晰度和有效性
        
        Args:
            prompt: 提示词
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 规则应用结果
        """
        original = prompt
        modified = False
        impact = "none"
        
        # 检查是否有明确的指令动词
        instruction_verbs = ["analyze", "summarize", "explain", "describe", "list", 
                           "compare", "contrast", "evaluate", "identify", "generate"]
        has_instruction_verb = any(verb in prompt.lower() for verb in instruction_verbs)
        
        # 检查是否有与内容类型相关的特定指令
        content_type = context.get("content_type", "unknown")
        has_specific_instruction = False
        
        if content_type == "formula":
            has_specific_instruction = any(kw in prompt.lower() for kw in ["formula", "equation", "variables", "solve"])
        elif content_type in ["chart", "diagram"]:
            has_specific_instruction = any(kw in prompt.lower() for kw in ["trend", "pattern", "interpret", "diagram", "chart"])
        elif content_type == "table":
            has_specific_instruction = any(kw in prompt.lower() for kw in ["table", "data", "row", "column"])
        
        # 添加明确的指令动词（如果需要）
        if not has_instruction_verb:
            # 在适当位置添加指令动词
            instruction_added = False
            
            if "请" in prompt:
                # 中文提示词
                index = prompt.find("请") + 1
                segments = list(prompt)
                segments.insert(index, "详细分析并")
                prompt = ''.join(segments)
                instruction_added = True
            elif re.search(r'\bplease\b', prompt, re.IGNORECASE):
                # 英文提示词
                prompt = re.sub(r'\bplease\b', 'please analyze and', prompt, count=1, flags=re.IGNORECASE)
                instruction_added = True
            else:
                # 在开头添加
                prompt = f"Analyze and explain the following content:\n\n{prompt}"
                instruction_added = True
                
            if instruction_added:
                modified = True
                impact = "medium"
        
        # 添加与内容类型相关的特定指令（如果需要）
        if not has_specific_instruction:
            specific_instruction = ""
            
            if content_type == "formula":
                specific_instruction = "\n\nFor each formula, explain its purpose, the meaning of each variable, and how it is applied."
            elif content_type == "chart":
                specific_instruction = "\n\nInterpret the chart by identifying trends, patterns, and key data points. Explain what insights can be drawn from this visualization."
            elif content_type == "diagram":
                specific_instruction = "\n\nExplain what this diagram represents, identify its key components, and describe the relationships or processes it illustrates."
            elif content_type == "table":
                specific_instruction = "\n\nAnalyze the table data, identify significant values, and explain what patterns or conclusions can be drawn from this information."
            
            if specific_instruction:
                prompt += specific_instruction
                modified = True
                impact = "high"
        
        return {
            "prompt": prompt,
            "modified": modified,
            "impact": impact,
            "original": original
        }
    
    def _rule_remove_redundancy(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        规则：删除提示词中的冗余和重复内容
        
        Args:
            prompt: 提示词
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 规则应用结果
        """
        original = prompt
        modified = False
        impact = "none"
        
        # 拆分成段落
        paragraphs = prompt.split('\n\n')
        
        if len(paragraphs) <= 1:
            # 如果没有明确分段，则不应用此规则
            return {
                "prompt": prompt,
                "modified": False,
                "impact": "none",
                "original": original
            }
        
        # 检测重复的段落
        unique_paragraphs = []
        for para in paragraphs:
            # 计算段落的"特征签名"（简化内容）
            simplified = re.sub(r'\s+', ' ', para.lower()).strip()
            
            # 检查是否有非常相似的段落已经存在
            is_duplicate = False
            for existing in unique_paragraphs:
                existing_simplified = re.sub(r'\s+', ' ', existing.lower()).strip()
                
                # 如果两段内容非常相似
                if simplified == existing_simplified:
                    is_duplicate = True
                    break
                # 或者一段是另一段的严格子集
                elif simplified in existing_simplified or existing_simplified in simplified:
                    # 保留较长的段落
                    if len(para) > len(existing):
                        unique_paragraphs.remove(existing)
                        unique_paragraphs.append(para)
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_paragraphs.append(para)
        
        # 检查是否删除了任何段落
        if len(unique_paragraphs) < len(paragraphs):
            prompt = '\n\n'.join(unique_paragraphs)
            modified = True
            impact = "medium"
        
        # 检查段落内部的冗余短语
        redundant_phrases = [
            r'As (I|we) (mentioned|said|noted) (earlier|before|previously)',
            r'As you can see',
            r'It is (important|worth) (to note|noting) that',
            r'Please note that',
            r'I would like to (point out|emphasize) that',
            r'Let me (remind you|reiterate) that'
        ]
        
        for i, para in enumerate(unique_paragraphs):
            original_para = para
            for phrase in redundant_phrases:
                para = re.sub(phrase, '', para, flags=re.IGNORECASE)
            
            # 如果删除了冗余短语，更新段落
            if para != original_para:
                unique_paragraphs[i] = re.sub(r'\s+', ' ', para).strip()
                modified = True
                if impact == "none":
                    impact = "low"
        
        if modified:
            prompt = '\n\n'.join(unique_paragraphs)
        
        return {
            "prompt": prompt,
            "modified": modified,
            "impact": impact,
            "original": original
        }
    
    def _calculate_specificity(self, prompt: str) -> float:
        """
        计算提示词的具体性得分
        
        Args:
            prompt: 提示词
            
        Returns:
            float: 具体性得分 (0-1)
        """
        # 计算具体细节指标
        specificity_indicators = [
            # 具体的指令动词
            r'\b(analyze|explain|describe|list|identify|categorize|compare|evaluate)\b',
            # 具体的格式指导
            r'\b(format|structure|organize|section|heading|bullet point|numbered list)\b',
            # 具体的定量指标
            r'\b(\d+\s+(point|section|example|reason|step)s?)\b',
            # 特定领域术语
            r'\b(formula|equation|variable|chart|diagram|trend|pattern|data|table|row|column)\b'
        ]
        
        # 计算匹配的指标数
        score = 0
        for indicator in specificity_indicators:
            matches = re.findall(indicator, prompt, re.IGNORECASE)
            if matches:
                score += len(matches) / 2  # 避免单一类别过多导致分数过高
        
        # 标准化得分 (0-1范围)
        normalized_score = min(score / 5, 1.0)
        
        return normalized_score


# 创建单例实例，方便直接导入使用
prompt_optimizer = PromptOptimizer()