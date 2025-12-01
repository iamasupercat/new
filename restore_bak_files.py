#!/usr/bin/env python3
"""
datasets 폴더 하위의 모든 .bak 파일을 원본 .txt로 복구하는 스크립트
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def find_all_bak_files(root_dir):
    """지정된 디렉토리 하위의 모든 .bak 파일 찾기"""
    root_path = Path(root_dir)
    bak_files = list(root_path.rglob("*.bak"))
    return bak_files

def restore_bak_file(bak_path, overwrite=False):
    """.bak 파일을 원본 .txt로 복구
    
    Args:
        bak_path: .bak 파일 경로
        overwrite: 원본 파일이 존재할 때 덮어쓸지 여부
    """
    bak_path = Path(bak_path)
    
    # .bak 파일이 .txt.bak 형태인지 확인
    if not bak_path.suffix == '.bak':
        return False, "파일 확장자가 .bak이 아닙니다."
    
    # 원본 파일 경로 생성 (filename.txt.bak -> filename.txt)
    original_path = bak_path.with_suffix('')
    
    # 원본 파일이 이미 존재하는 경우
    if original_path.exists():
        if not overwrite:
            return False, f"원본 파일이 이미 존재합니다: {original_path}"
        # 덮어쓰기 모드: 기존 원본 파일 삭제 후 .bak 파일을 원본으로 이동
        try:
            original_path.unlink()  # 기존 원본 파일 삭제
            shutil.move(str(bak_path), str(original_path))
            return True, "덮어쓰기 완료"
        except Exception as e:
            return False, f"덮어쓰기 실패: {e}"
    
    # 원본 파일이 없는 경우: .bak 파일을 원본으로 복구
    try:
        shutil.move(str(bak_path), str(original_path))
        return True, None
    except Exception as e:
        return False, f"복구 실패: {e}"

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='.bak 파일을 원본 .txt로 복구')
    parser.add_argument('--overwrite', action='store_true', 
                       help='원본 파일이 존재할 때 덮어쓰기 (기본값: False)')
    parser.add_argument('--dir', type=str, default='/home/work/datasets',
                       help='검색할 디렉토리 (기본값: /home/work/datasets)')
    args = parser.parse_args()
    
    datasets_dir = args.dir
    
    print(f"\n{'='*60}")
    print(f"🔄 .bak 파일 복구 시작")
    print(f"{'='*60}\n")
    print(f"📁 검색 디렉토리: {datasets_dir}")
    print(f"⚙️  덮어쓰기 모드: {'활성화' if args.overwrite else '비활성화'}\n")
    
    # 모든 .bak 파일 찾기
    print("🔍 .bak 파일 검색 중...")
    bak_files = find_all_bak_files(datasets_dir)
    
    if not bak_files:
        print("✓ 복구할 .bak 파일이 없습니다.")
        return
    
    print(f"📋 발견된 .bak 파일: {len(bak_files)}개\n")
    
    # 각 .bak 파일 복구
    restored_count = 0
    overwritten_count = 0
    skipped_count = 0
    error_count = 0
    
    iterator = tqdm(bak_files, desc="복구 중") if 'tqdm' in globals() else bak_files
    
    for bak_file in iterator:
        success, error_msg = restore_bak_file(bak_file, overwrite=args.overwrite)
        
        if success:
            if error_msg and "덮어쓰기" in error_msg:
                overwritten_count += 1
            else:
                restored_count += 1
        elif error_msg and "이미 존재" in error_msg:
            skipped_count += 1
        else:
            error_count += 1
            if error_msg:
                print(f"❌ {bak_file}: {error_msg}")
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"✅ 복구 완료!")
    print(f"{'='*60}")
    print(f"📊 결과:")
    print(f"  - 복구 성공: {restored_count}개")
    if args.overwrite:
        print(f"  - 덮어쓰기: {overwritten_count}개")
    print(f"  - 스킵 (원본 존재): {skipped_count}개")
    print(f"  - 오류: {error_count}개")
    print(f"  - 총 파일: {len(bak_files)}개")

if __name__ == "__main__":
    main()

